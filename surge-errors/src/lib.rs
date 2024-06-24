use error_tree::*;

error_tree!{

    pub enum ConvertError {
        Default,
    }

    pub enum AlignmentError {
        SrcPtr { idx: usize, required_align: usize },
        DstPtr { idx: usize, required_align: usize },
    }
}
