//! Display the input and output structure of an ONNX model.
use ndarray::Axis;
use onnxruntime::environment;
use onnxruntime::tensor::OrtOwnedTensor;
use tokenizers::tokenizer::{Result, Tokenizer};

pub fn embedding(input: &'static str) -> Result<Vec<f32>> {
    let tokenizer_path = "onnx/tokenizer.json";
    let model_path = "onnx/model.onnx";

    // Load the tokenizer and encode the input
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let encoding = tokenizer.encode(input, true)?;

    // Initialize the ONNX runtime environment and load the model
    let environment = environment::Environment::builder()
        .with_name("onnx metadata")
        .with_log_level(onnxruntime::LoggingLevel::Verbose)
        .build()?;

    // Create a new session with optimizations
    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(onnxruntime::GraphOptimizationLevel::Basic)?
        .with_model_from_file(model_path)?;

    // Convert the encoded input to the format expected by the ONNX model
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let attention_mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&x| x as i64)
        .collect();

    // Prepare the input tensors
    let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
    let input_ids_array = ndarray::Array::from_shape_vec((1, input_ids.len()), input_ids).unwrap();
    let attention_mask_array =
        ndarray::Array::from_shape_vec((1, attention_mask.len()), attention_mask).unwrap();
    let token_type_ids_array =
        ndarray::Array::from_shape_vec((1, token_type_ids.len()), token_type_ids).unwrap();

    // Run the model on the input tensors and retrieve the output
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![
        input_ids_array.clone(),
        attention_mask_array.clone(),
        token_type_ids_array,
    ])?;

    // Extract and expand the token embeddings
    let token_embeddings = outputs[0].to_owned();
    let input_mask_expanded = attention_mask_array
        .clone()
        .insert_axis(Axis(2))
        .broadcast((
            attention_mask_array.nrows(),
            attention_mask_array.ncols(),
            token_embeddings.shape()[2],
        ))
        .unwrap()
        .mapv(|x| x as f32);

    // Calculate the sentence embeddings from the output
    let token_masked_sum = (&token_embeddings * &input_mask_expanded).sum_axis(Axis(1));
    let mask_sum = input_mask_expanded.sum_axis(Axis(1)).mapv(|x| x.max(1e-9));
    let mean_pooling = token_masked_sum / mask_sum;
    let l2_norm = mean_pooling.mapv(|x| x.powi(2)).sum().sqrt();
    let sentence_embeddings = mean_pooling / l2_norm;

    // Convert to Vec<f32>
    let mut vec = Vec::new();
    for i in sentence_embeddings.iter() {
        vec.push(*i);
    }
    Ok(vec)
}
