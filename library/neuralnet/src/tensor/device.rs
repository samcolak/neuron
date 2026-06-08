#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendTrainingCapabilities {
    pub native_forward_execution: bool,
    pub native_backward_execution: bool,
    pub host_materializes_gradients: bool,
    pub device_resident_parameter_state: bool,
    pub device_side_parameter_updates: bool,
}

impl BackendTrainingCapabilities {
    pub const fn host_only() -> Self {
        Self {
            native_forward_execution: false,
            native_backward_execution: false,
            host_materializes_gradients: true,
            device_resident_parameter_state: false,
            device_side_parameter_updates: false,
        }
    }

    pub const fn native_compute_host_training() -> Self {
        Self {
            native_forward_execution: true,
            native_backward_execution: true,
            host_materializes_gradients: true,
            device_resident_parameter_state: false,
            device_side_parameter_updates: false,
        }
    }

    pub const fn fully_device_resident_training() -> Self {
        Self {
            native_forward_execution: true,
            native_backward_execution: true,
            host_materializes_gradients: false,
            device_resident_parameter_state: true,
            device_side_parameter_updates: true,
        }
    }

    pub const fn supports_device_resident_training(&self) -> bool {
        self.device_resident_parameter_state && self.device_side_parameter_updates
    }
}