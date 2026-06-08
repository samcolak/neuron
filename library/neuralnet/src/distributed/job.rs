use serde::{Deserialize, Serialize};

use crate::distributed::TensorResidency;
use crate::tensor::tensor4d::Tensor4D;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteTensorRef {
    pub peer_id: String,
    pub tensor_id: String,
    pub shape: (usize, usize, usize, usize),
    pub residency: TensorResidency,
    pub version: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RemoteTensorOp {
    Conv2dValid {
        input: Tensor4D,
        kernels: Tensor4D,
        bias: Option<Vec<f32>>,
        stride_h: usize,
        stride_w: usize,
    },
    MaxPool2d {
        input: Tensor4D,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    },
    GlobalAveragePool2d {
        input: Tensor4D,
    },
    Relu {
        input: Tensor4D,
    },
    ConvReluMaxPool2dValid {
        input: Tensor4D,
        kernels: Tensor4D,
        bias: Option<Vec<f32>>,
        conv_stride_h: usize,
        conv_stride_w: usize,
        pool_window_h: usize,
        pool_window_w: usize,
        pool_stride_h: usize,
        pool_stride_w: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteConvBlockDescriptor {
    pub kernels: Tensor4D,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteTensorOpRequest {
    pub operation: RemoteTensorOp,
    pub parameter_version: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteFeatureStackForwardRequest {
    pub input: Tensor4D,
    pub blocks: Vec<RemoteConvBlockDescriptor>,
    pub parameter_version: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistributedTensorJob {
    TensorOp(RemoteTensorOpRequest),
    FeatureStackForward(RemoteFeatureStackForwardRequest),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistributedTensorJobResult {
    Tensor(Tensor4D),
    FeatureBatch(Vec<Vec<f32>>),
    RemoteTensor(RemoteTensorRef),
}