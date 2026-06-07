use neuralnet::helpers::multimodal_controller::MultiModalInput;
use neuralnet::tensor::adapters::{
    image_bytes_to_tensor_nchw,
    image_bytes_to_tensor_nchw_resized_with_channels,
    image_bytes_to_tensor_nchw_with_channels,
    multimodal_input_to_tensor_nchw,
    multimodal_input_to_tensor_nchw_resized,
    TensorAdapterError,
};

#[test]
fn multimodal_image_bytes_maps_to_single_channel_tensor() {
    let input = MultiModalInput::ImageBytes(vec![0u8, 64u8, 128u8, 255u8]);

    let tensor = multimodal_input_to_tensor_nchw(&input, 2, 2, true)
        .unwrap_or_else(|_| panic!("image bytes should map to NCHW tensor"));

    assert_eq!(tensor.shape(), (1, 1, 2, 2));
    assert_eq!(tensor.get(0, 0, 0, 0), Ok(0.0));
    assert_eq!(tensor.get(0, 0, 1, 1), Ok(1.0));
}

#[test]
fn multimodal_image_bytes_maps_to_resized_tensor() {
    let input = MultiModalInput::ImageBytes(vec![0u8, 255u8, 255u8, 0u8]);

    let tensor = multimodal_input_to_tensor_nchw_resized(&input, 2, 2, 4, 4, false)
        .unwrap_or_else(|_| panic!("resized multimodal image mapping should succeed"));

    assert_eq!(tensor.shape(), (1, 1, 4, 4));
    assert_eq!(tensor.get(0, 0, 0, 0), Ok(0.0));
    assert_eq!(tensor.get(0, 0, 0, 3), Ok(255.0));
}

#[test]
fn tensor_variants_single_vs_rgb_have_expected_channel_shapes() {
    let grayscale = image_bytes_to_tensor_nchw(&[10u8, 20u8, 30u8, 40u8], 2, 2, false)
        .unwrap_or_else(|_| panic!("grayscale conversion should succeed"));

    let rgb_interleaved = image_bytes_to_tensor_nchw_with_channels(
        &[
            10u8, 11u8, 12u8,
            20u8, 21u8, 22u8,
            30u8, 31u8, 32u8,
            40u8, 41u8, 42u8,
        ],
        2,
        2,
        3,
        false,
    )
    .unwrap_or_else(|_| panic!("rgb conversion should succeed"));

    assert_eq!(grayscale.shape(), (1, 1, 2, 2));
    assert_eq!(rgb_interleaved.shape(), (1, 3, 2, 2));

    assert_eq!(rgb_interleaved.get(0, 0, 0, 0), Ok(10.0));
    assert_eq!(rgb_interleaved.get(0, 1, 0, 0), Ok(11.0));
    assert_eq!(rgb_interleaved.get(0, 2, 0, 0), Ok(12.0));
}

#[test]
fn tensor_variant_rgb_resize_preserves_channel_order() {
    let rgb = vec![
        5u8, 6u8, 7u8,
        15u8, 16u8, 17u8,
        25u8, 26u8, 27u8,
        35u8, 36u8, 37u8,
    ];

    let tensor = image_bytes_to_tensor_nchw_resized_with_channels(
        rgb.as_slice(),
        2,
        2,
        3,
        1,
        1,
        false,
    )
    .unwrap_or_else(|_| panic!("rgb resize conversion should succeed"));

    assert_eq!(tensor.shape(), (1, 3, 1, 1));
    assert_eq!(tensor.get(0, 0, 0, 0), Ok(5.0));
    assert_eq!(tensor.get(0, 1, 0, 0), Ok(6.0));
    assert_eq!(tensor.get(0, 2, 0, 0), Ok(7.0));
}

#[test]
fn multimodal_non_image_variants_are_rejected_for_tensor_adaptation() {
    let text = MultiModalInput::Text("cat on mat".to_string());
    let features = MultiModalInput::FeatureTokens {
        modality: "vision".to_string(),
        tokens: vec!["edge:04".to_string()],
    };

    let text_result = multimodal_input_to_tensor_nchw(&text, 1, 1, false);
    let features_result = multimodal_input_to_tensor_nchw(&features, 1, 1, false);

    assert_eq!(text_result, Err(TensorAdapterError::UnsupportedInputType));
    assert_eq!(features_result, Err(TensorAdapterError::UnsupportedInputType));
}
