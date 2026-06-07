use neuralnet::helpers::multimodal_controller::MultiModalInput;
use neuralnet::tensor::tensor4d::Tensor4D;
use neuralnet::tensor::adapters::{
    image_bytes_to_tensor_nchw,
    image_bytes_to_tensor_nchw_resized_with_channels,
    image_bytes_to_tensor_nchw_with_channels,
    multimodal_input_to_tensor_nchw,
    multimodal_input_to_tensor_nchw_resized,
};
use std::time::Instant;

fn compare_tensors(left: &Tensor4D, right: &Tensor4D) -> (usize, f32, f32) {
    let mut mismatch_count = 0usize;
    let mut max_abs_delta = 0.0f32;
    let mut mean_abs_delta = 0.0f32;

    for (l, r) in left.as_slice().iter().zip(right.as_slice().iter()) {
        let delta = (*l - *r).abs();
        if delta > 0.0 {
            mismatch_count += 1;
        }
        if delta > max_abs_delta {
            max_abs_delta = delta;
        }
        mean_abs_delta += delta;
    }

    if !left.is_empty() {
        mean_abs_delta /= left.len() as f32;
    }

    (mismatch_count, max_abs_delta, mean_abs_delta)
}

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

    println!("  step 6: explicit tensor comparison metrics");
    let single = Tensor4D::from_vec(1, 1, 2, 2, vec![1.0, 2.0, 3.0, 4.0])
        .unwrap_or_else(|_| panic!("single tensor should be valid"));
    let multi = Tensor4D::from_vec(
        1,
        2,
        2,
        2,
        vec![
            1.0, 2.0, 3.0, 4.0, // channel 0
            0.0, 0.0, 0.0, 0.0, // channel 1
        ],
    )
    .unwrap_or_else(|_| panic!("multi tensor should be valid"));

    let kernel_single = Tensor4D::from_vec(1, 1, 1, 1, vec![2.0])
        .unwrap_or_else(|_| panic!("single kernel should be valid"));
    let kernel_multi = Tensor4D::from_vec(1, 2, 1, 1, vec![2.0, 0.0])
        .unwrap_or_else(|_| panic!("multi kernel should be valid"));

    let start_single = Instant::now();
    let out_single = single
        .conv2d_valid(&kernel_single, None, 1, 1)
        .unwrap_or_else(|_| panic!("single conv should succeed"));
    let single_us = start_single.elapsed().as_micros();

    let start_multi = Instant::now();
    let out_multi = multi
        .conv2d_valid(&kernel_multi, None, 1, 1)
        .unwrap_or_else(|_| panic!("multi conv should succeed"));
    let multi_us = start_multi.elapsed().as_micros();

    let (mismatch_count, max_abs_delta, mean_abs_delta) = compare_tensors(&out_single, &out_multi);
    let numerically_equivalent = mismatch_count == 0;

    println!(
        "    single_vs_multi_zero_channel: shape={:?} mismatches={} max_abs_delta={:.6} mean_abs_delta={:.6}",
        out_single.shape(),
        mismatch_count,
        max_abs_delta,
        mean_abs_delta
    );
    println!(
        "    single_conv_us={} multi_conv_us={} equivalent={}",
        single_us,
        multi_us,
        numerically_equivalent
    );

    let single_signal = Tensor4D::from_vec(1, 1, 1, 1, vec![2.0])
        .unwrap_or_else(|_| panic!("single signal tensor should be valid"));
    let multi_signal = Tensor4D::from_vec(1, 2, 1, 1, vec![2.0, 5.0])
        .unwrap_or_else(|_| panic!("multi signal tensor should be valid"));
    let kernel_single_signal = Tensor4D::from_vec(1, 1, 1, 1, vec![3.0])
        .unwrap_or_else(|_| panic!("single signal kernel should be valid"));
    let kernel_multi_signal = Tensor4D::from_vec(1, 2, 1, 1, vec![3.0, 1.0])
        .unwrap_or_else(|_| panic!("multi signal kernel should be valid"));

    let out_single_signal = single_signal
        .conv2d_valid(&kernel_single_signal, None, 1, 1)
        .unwrap_or_else(|_| panic!("single signal conv should succeed"));
    let out_multi_signal = multi_signal
        .conv2d_valid(&kernel_multi_signal, None, 1, 1)
        .unwrap_or_else(|_| panic!("multi signal conv should succeed"));

    let baseline = out_single_signal.get(0, 0, 0, 0).unwrap_or(0.0);
    let enriched = out_multi_signal.get(0, 0, 0, 0).unwrap_or(0.0);
    let uplift = enriched - baseline;

    println!(
        "    extra_channel_signal: baseline={:.3} enriched={:.3} uplift={:.3}",
        baseline,
        enriched,
        uplift
    );
}
