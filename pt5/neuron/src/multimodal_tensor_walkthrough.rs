use neuralnet::helpers::multimodal_controller::MultiModalInput;
use neuralnet::tensor::adapters::{
    image_bytes_to_tensor_nchw,
    image_bytes_to_tensor_nchw_resized_with_channels,
    image_bytes_to_tensor_nchw_with_channels,
    multimodal_input_to_tensor_nchw,
    multimodal_input_to_tensor_nchw_resized,
};

pub fn run_multimodal_tensor_walkthrough() {
    println!("\nMultimodal tensor walkthrough");

    println!("  step 1: single-channel image bytes -> NCHW tensor");
    let grayscale = vec![0u8, 64u8, 128u8, 255u8];
    let single = image_bytes_to_tensor_nchw(grayscale.as_slice(), 2, 2, true)
        .unwrap_or_else(|_| panic!("single-channel conversion should succeed"));
    println!(
        "    shape={:?} first={:?} last={:?}",
        single.shape(),
        single.get(0, 0, 0, 0),
        single.get(0, 0, 1, 1)
    );

    println!("  step 2: RGB interleaved image bytes -> 3-channel NCHW tensor");
    let rgb = vec![
        10u8, 11u8, 12u8,
        20u8, 21u8, 22u8,
        30u8, 31u8, 32u8,
        40u8, 41u8, 42u8,
    ];
    let rgb_tensor = image_bytes_to_tensor_nchw_with_channels(rgb.as_slice(), 2, 2, 3, false)
        .unwrap_or_else(|_| panic!("rgb conversion should succeed"));
    println!(
        "    shape={:?} c0[0,0]={:?} c1[0,0]={:?} c2[0,0]={:?}",
        rgb_tensor.shape(),
        rgb_tensor.get(0, 0, 0, 0),
        rgb_tensor.get(0, 1, 0, 0),
        rgb_tensor.get(0, 2, 0, 0)
    );

    println!("  step 3: multimodal image input -> resized tensor");
    let mm_image = MultiModalInput::ImageBytes(vec![0u8, 255u8, 255u8, 0u8]);
    let resized = multimodal_input_to_tensor_nchw_resized(&mm_image, 2, 2, 4, 4, false)
        .unwrap_or_else(|_| panic!("resized multimodal conversion should succeed"));
    println!(
        "    shape={:?} top_left={:?} top_right={:?}",
        resized.shape(),
        resized.get(0, 0, 0, 0),
        resized.get(0, 0, 0, 3)
    );

    println!("  step 4: direct RGB resize with channels preserved");
    let resized_rgb = image_bytes_to_tensor_nchw_resized_with_channels(
        rgb.as_slice(),
        2,
        2,
        3,
        1,
        1,
        false,
    )
    .unwrap_or_else(|_| panic!("rgb resized conversion should succeed"));
    println!(
        "    shape={:?} c0={:?} c1={:?} c2={:?}",
        resized_rgb.shape(),
        resized_rgb.get(0, 0, 0, 0),
        resized_rgb.get(0, 1, 0, 0),
        resized_rgb.get(0, 2, 0, 0)
    );

    println!("  step 5: non-image multimodal variants are rejected");
    let text = MultiModalInput::Text("cat on mat".to_string());
    let text_result = multimodal_input_to_tensor_nchw(&text, 1, 1, false);
    println!("    text adaptation result={:?}", text_result);
}
