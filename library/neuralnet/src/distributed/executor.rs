use crate::distributed::{
    DistributedExecutionPolicy,
    DistributedExecutorCapabilities,
    DistributedTensorError,
    DistributedTensorJob,
    DistributedTensorJobResult,
    RemotePeerDescriptor,
};

pub trait DistributedTensorExecutor: Send + Sync {
    
    fn name(&self) -> &'static str;

    fn policy(&self) -> &DistributedExecutionPolicy;

    fn capabilities(&self) -> DistributedExecutorCapabilities;

    fn execute(
        &self,
        peer: &RemotePeerDescriptor,
        job: DistributedTensorJob,
    ) -> Result<DistributedTensorJobResult, DistributedTensorError>;

}