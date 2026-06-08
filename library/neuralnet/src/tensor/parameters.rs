use crate::tensor::backend::{active_backend_name, active_backend_training_capabilities};
#[cfg(feature = "offloading-mlx")]
use crate::tensor::offloading::mlx_backend::{
    mlx_add_tensor_in_place,
    mlx_apply_sgd_update_from_array_in_place,
    mlx_owned_array_from_tensor,
    mlx_owned_array_to_tensor,
    mlx_zero_in_place,
    MlxOwnedArray,
};
use crate::tensor::tensor4d::{Tensor4D, TensorError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterResidency {
    HostBacked,
    BackendMirrored,
    DeviceResident,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterBackendKind {
    Host,
    Mlx,
    Cuda,
}

#[derive(Debug, Clone, PartialEq)]
struct HostConvParameterState {
    kernels: Tensor4D,
    bias: Vec<f32>,
    grad_accum_kernel: Tensor4D,
    grad_accum_bias: Vec<f32>,
}

impl HostConvParameterState {
    fn new(kernels: Tensor4D, bias: Vec<f32>) -> Self {
        let (n, c, h, w) = kernels.shape();
        Self {
            kernels,
            grad_accum_kernel: Tensor4D::zeros(n, c, h, w),
            grad_accum_bias: vec![0.0; bias.len()],
            bias,
        }
    }
}

#[derive(Debug)]
struct MlxMirroredConvParameterState {
    host: HostConvParameterState,
    #[cfg(feature = "offloading-mlx")]
    device_kernels: Option<MlxOwnedArray>,
    #[cfg(feature = "offloading-mlx")]
    device_bias: Option<MlxOwnedArray>,
    #[cfg(feature = "offloading-mlx")]
    device_grad_kernels: Option<MlxOwnedArray>,
    #[cfg(feature = "offloading-mlx")]
    device_grad_bias: Option<MlxOwnedArray>,
    host_dirty: bool,
    device_dirty: bool,
}

impl MlxMirroredConvParameterState {
    fn new(kernels: Tensor4D, bias: Vec<f32>) -> Self {
        let host = HostConvParameterState::new(kernels, bias);
        #[cfg(feature = "offloading-mlx")]
        let (device_kernels, device_bias, device_grad_kernels, device_grad_bias) = (
            mlx_owned_array_from_tensor(&host.kernels).ok(),
            Tensor4D::from_vec(1, host.bias.len(), 1, 1, host.bias.clone())
                .ok()
                .and_then(|tensor| mlx_owned_array_from_tensor(&tensor).ok()),
            mlx_owned_array_from_tensor(&host.grad_accum_kernel).ok(),
            Tensor4D::from_vec(1, host.grad_accum_bias.len(), 1, 1, host.grad_accum_bias.clone())
                .ok()
                .and_then(|tensor| mlx_owned_array_from_tensor(&tensor).ok()),
        );

        Self {
            host,
            #[cfg(feature = "offloading-mlx")]
            device_kernels,
            #[cfg(feature = "offloading-mlx")]
            device_bias,
            #[cfg(feature = "offloading-mlx")]
            device_grad_kernels,
            #[cfg(feature = "offloading-mlx")]
            device_grad_bias,
            host_dirty: false,
            device_dirty: false,
        }
    }

    fn sync_device_from_host(&mut self) {
        #[cfg(feature = "offloading-mlx")]
        {
            self.device_kernels = mlx_owned_array_from_tensor(&self.host.kernels).ok();
            self.device_bias = Tensor4D::from_vec(1, self.host.bias.len(), 1, 1, self.host.bias.clone())
                .ok()
                .and_then(|tensor| mlx_owned_array_from_tensor(&tensor).ok());
        }
        self.host_dirty = false;
    }

    fn sync_host_from_device(&mut self) {
        #[cfg(feature = "offloading-mlx")]
        {
            if let Some(device_kernels) = self.device_kernels.as_ref()
                && let Ok(kernels) = mlx_owned_array_to_tensor(device_kernels)
            {
                self.host.kernels = kernels;
            }

            if let Some(device_bias) = self.device_bias.as_ref()
                && let Ok(bias_tensor) = mlx_owned_array_to_tensor(device_bias)
            {
                self.host.bias = bias_tensor.first_sample_features();
            }
        }
        self.device_dirty = false;
    }

    fn apply_sgd_update(&mut self, learning_rate: f32, batch_size: f32) -> Result<(), TensorError> {
        apply_sgd_update_single(
            &mut self.host.kernels,
            self.host.bias.as_mut_slice(),
            &self.host.grad_accum_kernel,
            self.host.grad_accum_bias.as_slice(),
            learning_rate,
            batch_size,
        )?;

        #[cfg(feature = "offloading-mlx")]
        {
            if let (Some(device_kernels), Some(device_grad_kernels)) =
                (self.device_kernels.as_mut(), self.device_grad_kernels.as_ref())
            {
                mlx_apply_sgd_update_from_array_in_place(
                    device_kernels,
                    device_grad_kernels,
                    learning_rate,
                    batch_size,
                )?;
            }

            if let (Some(device_bias), Some(device_grad_bias)) =
                (self.device_bias.as_mut(), self.device_grad_bias.as_ref())
            {
                mlx_apply_sgd_update_from_array_in_place(
                    device_bias,
                    device_grad_bias,
                    learning_rate,
                    batch_size,
                )?;
            }

            if let Some(device_grad_kernels) = self.device_grad_kernels.as_mut() {
                mlx_zero_in_place(device_grad_kernels)?;
            }
            if let Some(device_grad_bias) = self.device_grad_bias.as_mut() {
                mlx_zero_in_place(device_grad_bias)?;
            }
        }

        self.host_dirty = false;
        self.device_dirty = false;
        self.host.grad_accum_kernel.fill(0.0);
        self.host.grad_accum_bias.fill(0.0);
        Ok(())
    }
}

impl Clone for MlxMirroredConvParameterState {
    fn clone(&self) -> Self {
        Self::new(self.host.kernels.clone(), self.host.bias.clone())
    }
}

impl PartialEq for MlxMirroredConvParameterState {
    fn eq(&self, other: &Self) -> bool {
        self.host == other.host
            && self.host_dirty == other.host_dirty
            && self.device_dirty == other.device_dirty
    }
}

#[derive(Debug, Clone, PartialEq)]
struct CudaMirroredConvParameterState {
    host: HostConvParameterState,
}

impl CudaMirroredConvParameterState {
    fn new(kernels: Tensor4D, bias: Vec<f32>) -> Self {
        Self {
            host: HostConvParameterState::new(kernels, bias),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ConvParameterStorage {
    Host(HostConvParameterState),
    Mlx(MlxMirroredConvParameterState),
    Cuda(CudaMirroredConvParameterState),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvParameterState {
    backend_kind: ParameterBackendKind,
    residency: ParameterResidency,
    storage: ConvParameterStorage,
}

impl ConvParameterState {
    pub fn new(kernels: Tensor4D, bias: Vec<f32>) -> Self {
        Self::new_for_active_backend(kernels, bias)
    }

    pub fn new_for_active_backend(kernels: Tensor4D, bias: Vec<f32>) -> Self {
        let caps = active_backend_training_capabilities();
        if caps.supports_device_resident_training() {
            return Self::new_host_backed(kernels, bias);
        }

        match active_backend_name() {
            "mlx" => Self::new_mlx_mirrored(kernels, bias),
            "cuda" => Self::new_cuda_mirrored(kernels, bias),
            _ => Self::new_host_backed(kernels, bias),
        }
    }

    pub fn new_mlx_mirrored(kernels: Tensor4D, bias: Vec<f32>) -> Self {
        Self {
            backend_kind: ParameterBackendKind::Mlx,
            residency: ParameterResidency::BackendMirrored,
            storage: ConvParameterStorage::Mlx(MlxMirroredConvParameterState::new(kernels, bias)),
        }
    }

    pub fn new_cuda_mirrored(kernels: Tensor4D, bias: Vec<f32>) -> Self {
        Self {
            backend_kind: ParameterBackendKind::Cuda,
            residency: ParameterResidency::BackendMirrored,
            storage: ConvParameterStorage::Cuda(CudaMirroredConvParameterState::new(kernels, bias)),
        }
    }

    pub fn new_host_backed(kernels: Tensor4D, bias: Vec<f32>) -> Self {
        Self {
            backend_kind: ParameterBackendKind::Host,
            residency: ParameterResidency::HostBacked,
            storage: ConvParameterStorage::Host(HostConvParameterState::new(kernels, bias)),
        }
    }

    pub fn backend_kind(&self) -> ParameterBackendKind {
        self.backend_kind
    }

    #[cfg(feature = "offloading-mlx")]
    pub fn mlx_mirror_views(
        &self,
    ) -> Option<(
        &MlxOwnedArray,
        (usize, usize, usize, usize),
        &MlxOwnedArray,
    )> {
        match &self.storage {
            ConvParameterStorage::Mlx(state) if !state.host_dirty => Some((
                state.device_kernels.as_ref()?,
                state.host.kernels.shape(),
                state.device_bias.as_ref()?,
            )),
            _ => None,
        }
    }

    pub fn residency(&self) -> ParameterResidency {
        self.residency
    }

    pub fn kernels(&self) -> &Tensor4D {
        match &self.storage {
            ConvParameterStorage::Host(state) => &state.kernels,
            ConvParameterStorage::Mlx(state) => &state.host.kernels,
            ConvParameterStorage::Cuda(state) => &state.host.kernels,
        }
    }

    pub fn bias(&self) -> &[f32] {
        match &self.storage {
            ConvParameterStorage::Host(state) => state.bias.as_slice(),
            ConvParameterStorage::Mlx(state) => state.host.bias.as_slice(),
            ConvParameterStorage::Cuda(state) => state.host.bias.as_slice(),
        }
    }

    pub fn kernels_mut(&mut self) -> &mut Tensor4D {
        match &mut self.storage {
            ConvParameterStorage::Host(state) => &mut state.kernels,
            ConvParameterStorage::Mlx(state) => &mut state.host.kernels,
            ConvParameterStorage::Cuda(state) => &mut state.host.kernels,
        }
    }

    pub fn bias_mut(&mut self) -> &mut [f32] {
        match &mut self.storage {
            ConvParameterStorage::Host(state) => state.bias.as_mut_slice(),
            ConvParameterStorage::Mlx(state) => state.host.bias.as_mut_slice(),
            ConvParameterStorage::Cuda(state) => state.host.bias.as_mut_slice(),
        }
    }

    pub fn parameter_views_mut(&mut self) -> (&mut Tensor4D, &mut [f32]) {
        match &mut self.storage {
            ConvParameterStorage::Host(state) => (&mut state.kernels, state.bias.as_mut_slice()),
            ConvParameterStorage::Mlx(state) => {
                state.host_dirty = true;
                (&mut state.host.kernels, state.host.bias.as_mut_slice())
            }
            ConvParameterStorage::Cuda(state) => {
                (&mut state.host.kernels, state.host.bias.as_mut_slice())
            }
        }
    }

    pub fn reset_accumulated_gradients(&mut self) {
        match &mut self.storage {
            ConvParameterStorage::Host(state) => {
                state.grad_accum_kernel.fill(0.0);
                state.grad_accum_bias.fill(0.0);
            }
            ConvParameterStorage::Mlx(state) => {
                state.host.grad_accum_kernel.fill(0.0);
                state.host.grad_accum_bias.fill(0.0);
                #[cfg(feature = "offloading-mlx")]
                {
                    if let Some(device_grad_kernels) = state.device_grad_kernels.as_mut() {
                        let _ = mlx_zero_in_place(device_grad_kernels);
                    }
                    if let Some(device_grad_bias) = state.device_grad_bias.as_mut() {
                        let _ = mlx_zero_in_place(device_grad_bias);
                    }
                }
            }
            ConvParameterStorage::Cuda(state) => {
                state.host.grad_accum_kernel.fill(0.0);
                state.host.grad_accum_bias.fill(0.0);
            }
        }
    }

    pub fn accumulate_gradients(
        &mut self,
        kernel_grad: &Tensor4D,
        bias_grad: &[f32],
    ) -> Result<(), TensorError> {
        match &mut self.storage {
            ConvParameterStorage::Host(state) => {
                state.grad_accum_kernel.add_inplace(kernel_grad)?;

                if state.grad_accum_bias.len() != bias_grad.len() {
                    return Err(TensorError::ShapeMismatch {
                        expected: state.grad_accum_bias.len(),
                        actual: bias_grad.len(),
                    });
                }

                for (accum, grad) in state.grad_accum_bias.iter_mut().zip(bias_grad.iter()) {
                    *accum += *grad;
                }
            }
            ConvParameterStorage::Mlx(state) => {
                state.host.grad_accum_kernel.add_inplace(kernel_grad)?;

                if state.host.grad_accum_bias.len() != bias_grad.len() {
                    return Err(TensorError::ShapeMismatch {
                        expected: state.host.grad_accum_bias.len(),
                        actual: bias_grad.len(),
                    });
                }

                for (accum, grad) in state.host.grad_accum_bias.iter_mut().zip(bias_grad.iter()) {
                    *accum += *grad;
                }

                #[cfg(feature = "offloading-mlx")]
                {
                    if let Some(device_grad_kernels) = state.device_grad_kernels.as_mut() {
                        mlx_add_tensor_in_place(device_grad_kernels, kernel_grad)?;
                    }
                    if let Some(device_grad_bias) = state.device_grad_bias.as_mut() {
                        let bias_grad_tensor = Tensor4D::from_vec(
                            1,
                            bias_grad.len(),
                            1,
                            1,
                            bias_grad.to_vec(),
                        )?;
                        mlx_add_tensor_in_place(device_grad_bias, &bias_grad_tensor)?;
                    }
                }
            }
            ConvParameterStorage::Cuda(state) => {
                state.host.grad_accum_kernel.add_inplace(kernel_grad)?;

                if state.host.grad_accum_bias.len() != bias_grad.len() {
                    return Err(TensorError::ShapeMismatch {
                        expected: state.host.grad_accum_bias.len(),
                        actual: bias_grad.len(),
                    });
                }

                for (accum, grad) in state.host.grad_accum_bias.iter_mut().zip(bias_grad.iter()) {
                    *accum += *grad;
                }
            }
        }

        Ok(())
    }

    pub fn accumulated_kernel_grad(&self) -> &Tensor4D {
        match &self.storage {
            ConvParameterStorage::Host(state) => &state.grad_accum_kernel,
            ConvParameterStorage::Mlx(state) => &state.host.grad_accum_kernel,
            ConvParameterStorage::Cuda(state) => &state.host.grad_accum_kernel,
        }
    }

    pub fn accumulated_bias_grad(&self) -> &[f32] {
        match &self.storage {
            ConvParameterStorage::Host(state) => state.grad_accum_bias.as_slice(),
            ConvParameterStorage::Mlx(state) => state.host.grad_accum_bias.as_slice(),
            ConvParameterStorage::Cuda(state) => state.host.grad_accum_bias.as_slice(),
        }
    }

    pub fn accumulated_gradient_views(&self) -> (&Tensor4D, &[f32]) {
        match &self.storage {
            ConvParameterStorage::Host(state) => {
                (&state.grad_accum_kernel, state.grad_accum_bias.as_slice())
            }
            ConvParameterStorage::Mlx(state) => (&state.host.grad_accum_kernel, state.host.grad_accum_bias.as_slice()),
            ConvParameterStorage::Cuda(state) => (&state.host.grad_accum_kernel, state.host.grad_accum_bias.as_slice()),
        }
    }

    pub fn snapshot(&self) -> (Tensor4D, Vec<f32>) {
        match &self.storage {
            ConvParameterStorage::Host(state) => (state.kernels.clone(), state.bias.clone()),
            ConvParameterStorage::Mlx(state) => (state.host.kernels.clone(), state.host.bias.clone()),
            ConvParameterStorage::Cuda(state) => (state.host.kernels.clone(), state.host.bias.clone()),
        }
    }

    pub fn apply_sgd_update(
        &mut self,
        learning_rate: f32,
        batch_size: f32,
    ) -> Result<(), TensorError> {
        match &mut self.storage {
            ConvParameterStorage::Host(state) => {
                apply_sgd_update_single(
                    &mut state.kernels,
                    state.bias.as_mut_slice(),
                    &state.grad_accum_kernel,
                    state.grad_accum_bias.as_slice(),
                    learning_rate,
                    batch_size,
                )?;
                state.grad_accum_kernel.fill(0.0);
                state.grad_accum_bias.fill(0.0);
                Ok(())
            }
            ConvParameterStorage::Mlx(state) => state.apply_sgd_update(learning_rate, batch_size),
            ConvParameterStorage::Cuda(state) => {
                apply_sgd_update_single(
                    &mut state.host.kernels,
                    state.host.bias.as_mut_slice(),
                    &state.host.grad_accum_kernel,
                    state.host.grad_accum_bias.as_slice(),
                    learning_rate,
                    batch_size,
                )?;
                state.host.grad_accum_kernel.fill(0.0);
                state.host.grad_accum_bias.fill(0.0);
                Ok(())
            }
        }
    }

    pub fn sync_backend_mirror(&mut self) {
        match &mut self.storage {
            ConvParameterStorage::Host(_) => {}
            ConvParameterStorage::Mlx(state) => {
                if state.host_dirty {
                    state.sync_device_from_host();
                }
            }
            ConvParameterStorage::Cuda(_) => {}
        }
    }

    pub fn refresh_host_from_backend(&mut self) {
        match &mut self.storage {
            ConvParameterStorage::Host(_) => {}
            ConvParameterStorage::Mlx(state) => {
                if state.device_dirty {
                    state.sync_host_from_device();
                }
            }
            ConvParameterStorage::Cuda(_) => {}
        }
    }
}

fn apply_sgd_update_single(
    kernels: &mut Tensor4D,
    bias: &mut [f32],
    kernel_grad: &Tensor4D,
    bias_grad: &[f32],
    learning_rate: f32,
    batch_size: f32,
) -> Result<(), TensorError> {
    if kernels.shape() != kernel_grad.shape() {
        return Err(TensorError::IncompatibleShapes {
            left: kernels.shape(),
            right: kernel_grad.shape(),
        });
    }

    if bias.len() != bias_grad.len() {
        return Err(TensorError::ShapeMismatch {
            expected: bias.len(),
            actual: bias_grad.len(),
        });
    }

    let scale = if batch_size > 0.0 { 1.0 / batch_size } else { 1.0 };
    for (weight, grad) in kernels.as_mut_slice().iter_mut().zip(kernel_grad.as_slice().iter()) {
        *weight -= learning_rate * *grad * scale;
    }
    for (bias_value, grad) in bias.iter_mut().zip(bias_grad.iter()) {
        *bias_value -= learning_rate * *grad * scale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_parameter_state_defaults_to_host_backed_residency() {
        let state = ConvParameterState::new_host_backed(Tensor4D::zeros(2, 1, 3, 3), vec![0.0; 2]);
        assert_eq!(state.residency(), ParameterResidency::HostBacked);
        assert_eq!(state.backend_kind(), ParameterBackendKind::Host);
    }

    #[test]
    fn conv_parameter_state_can_represent_mlx_mirrored_storage() {
        let state = ConvParameterState::new_mlx_mirrored(Tensor4D::zeros(2, 1, 3, 3), vec![0.0; 2]);
        assert_eq!(state.residency(), ParameterResidency::BackendMirrored);
        assert_eq!(state.backend_kind(), ParameterBackendKind::Mlx);
    }
}