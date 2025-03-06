use anyhow::{anyhow, Context, Result};
use ndarray::{
  s, Array, Array1, Array2, Array3, Array4, Axis, IxDyn,
};
use ort::environment::Environment;
use ort::execution_providers::CPUExecutionProvider;
use ort::execution_providers::CUDAExecutionProvider;
use ort::execution_providers::CoreMLExecutionProvider;
use ort::execution_providers::ExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use ort::value::{Tensor, Value};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

enum InputMode {
  Language = 0,
  Vision = 1,
  Speech = 2,
  VisionSpeech = 3,
}

// Constants from the Python code
const IMAGE_SPECIAL_TOKEN_ID: i64 = 200010; // '<|endoftext10|>'
const AUDIO_SPECIAL_TOKEN_ID: i64 = 200011; // '<|endoftext11|>'

pub struct Phi4MMProcessor {
  vision_session: Session,
  speech_session: Session,
  embedding_session: Session,
  text_session: Session,
  tokenizer: Tokenizer,
  environment: Arc<Environment>,
}

impl Phi4MMProcessor {
  pub fn new<P: AsRef<Path>>(
    model_path: P,
    tokenizer_path: P,
    environment: Arc<Environment>,
  ) -> Result<Self> {
    // Load the ONNX models
    let vision_path =
      model_path.as_ref().join("phi-4-mm-vision.onnx");
    let speech_path =
      model_path.as_ref().join("phi-4-mm-speech.onnx");
    let embedding_path =
      model_path.as_ref().join("phi-4-mm-embedding.onnx");
    let text_path = model_path.as_ref().join("phi-4-mm-text.onnx");

    // Create sessions with appropriate optimization level
    let vision_session = SessionBuilder::new()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .commit_from_file(vision_path)?;

    let speech_session = SessionBuilder::new()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .commit_from_file(speech_path)?;

    let embedding_session = SessionBuilder::new()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .commit_from_file(embedding_path)?;

    dbg!("QQQ");
    let text_session = SessionBuilder::new()?
      .with_execution_providers([
        CUDAExecutionProvider::default().build(), // CPUExecutionProvider::default().build()
      ])?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .commit_from_file(text_path)?;

    // Load the tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)
      .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

    Ok(Self {
      vision_session,
      speech_session,
      embedding_session,
      text_session,
      tokenizer,
      environment,
    })
  }

  /// Process image input for the vision model
  fn process_image(
    &self,
    image: &[u8],
  ) -> Result<(Value, Value, Value)> {
    // This would need to preprocess the image according to the requirements of Phi-4's vision model
    // For simplicity, this is a placeholder
    // In a real implementation, we'd need to:
    // 1. Load and resize the image
    // 2. Normalize the pixel values
    // 3. Create the necessary tensors
    // 4. Create the attention mask

    // Example code for creating placeholder tensors
    let batch_size = 1;
    let max_num_crops = 5; // Adjust based on your needs
    let channels = 3;
    let height = 448;
    let width = 448;

    // Create image input tensor (batch_size, max_num_crops, channels, height, width)
    let pixel_values = Array::<f32, _>::zeros((
      batch_size,
      max_num_crops,
      channels,
      height,
      width,
    ));
    // In a real implementation, we'd fill this with the actual image data

    // Create attention mask - in a real implementation this would be based on the actual image content
    let attention_mask =
      Array::<f32, _>::ones((batch_size, max_num_crops, 32, 32));

    // Create image sizes - in a real implementation this would be the actual sizes
    let image_sizes = Array::<i64, _>::from_shape_vec(
      (batch_size, 2),
      vec![height as i64, width as i64],
    )?;

    let pixel_values_tensor =
      Tensor::from_array(pixel_values.into_dyn())?;
    let attention_mask_tensor =
      Tensor::from_array(attention_mask.into_dyn())?;
    let image_sizes_tensor =
      Tensor::from_array(image_sizes.into_dyn())?;

    // Convert to Value
    let pixel_values_value = pixel_values_tensor.into_dyn();
    let attention_mask_value = attention_mask_tensor.into_dyn();
    let image_sizes_value = image_sizes_tensor.into_dyn();

    Ok((
      pixel_values_value,
      attention_mask_value,
      image_sizes_value.into(),
    ))
  }

  /// Process audio input for the speech model
  fn process_audio(
    &self,
    audio_data: &[f32],
    sample_rate: i32,
  ) -> Result<(Value, Value, Value, Value)> {
    // This would need to preprocess the audio according to the requirements of Phi-4's speech model
    // For simplicity, this is a placeholder
    // In a real implementation, we'd need to:
    // 1. Extract the audio features (mel spectrograms)
    // 2. Create the necessary tensors
    // 3. Create the attention mask

    // Example code for creating placeholder tensors
    let batch_size = 1;
    let num_frames = 128; // Adjust based on your needs
    let feature_size = 80; // Mel spectrogram features

    // Create audio input tensor (batch_size, num_frames, feature_size)
    let audio_embeds =
      Array::<f32, _>::zeros((batch_size, num_frames, feature_size));
    // In a real implementation, we'd fill this with the actual audio features

    // Create audio attention mask
    let audio_attention_mask =
      Array::<i64, _>::ones((batch_size, num_frames));

    // Create audio embed sizes
    let audio_embed_sizes =
      Array::from_shape_vec((batch_size,), vec![num_frames as i64])?;

    // Create input mode - always Speech for this function
    let input_mode = Array::from_elem((), InputMode::Speech as i64);

    let audio_embeds_tensor =
      Tensor::from_array(audio_embeds.into_dyn())?;
    let audio_attention_mask_tensor =
      Tensor::from_array(audio_attention_mask.into_dyn())?;
    let audio_embed_sizes_tensor =
      Tensor::from_array(audio_embed_sizes.into_dyn())?;
    let input_mode_tensor =
      Tensor::from_array(input_mode.into_dyn())?;

    let audio_embeds_value = audio_embeds_tensor.into_dyn();
    let audio_attention_mask_value =
      audio_attention_mask_tensor.into_dyn();
    let audio_embed_sizes_value = audio_embed_sizes_tensor.into_dyn();
    let input_mode_value = input_mode_tensor.into_dyn();

    Ok((
      audio_embeds_value,
      audio_attention_mask_value,
      audio_embed_sizes_value.into(),
      input_mode_value.into(),
    ))
  }

  /// Process text input to create token IDs
  fn process_text(&self, text: &str) -> Result<Array2<i64>> {
    // Tokenize the text
    let encoding = self
      .tokenizer
      .encode(text, true)
      .map_err(|e| anyhow!("Failed to encode: {}", e))?;
    let input_ids = encoding
      .get_ids()
      .iter()
      .map(|&id| id as i64)
      .collect::<Vec<_>>();

    // Convert to 2D ndarray with batch size 1
    let seq_len = input_ids.len();
    Ok(Array2::from_shape_vec((1, seq_len), input_ids)?)
  }

  /// Run the combined embedding model to merge text, image, and audio inputs
  fn run_embedding_model(
    &self,
    input_ids: Array2<i64>,
    image_features: Option<Array2<f32>>,
    audio_features: Option<Array2<f32>>,
  ) -> Result<Array2<f32>> {
    // Prepare inputs for the embedding model
    let input_ids_tensor = Tensor::from_array(input_ids.into_dyn())?;
    let input_ids_value = input_ids_tensor.into_dyn();

    let image_features_value = match image_features {
      Some(features) => {
        let features_tensor =
          Tensor::from_array(features.into_dyn())?;
        features_tensor.into_dyn()
      }
      None => {
        let empty_features = Tensor::from_array(
          Array2::<f32>::zeros((0, 1152)).into_dyn(),
        )?;
        empty_features.into_dyn()
      }
    };

    let audio_features_value = match audio_features {
      Some(features) => {
        let features_tensor =
          Tensor::from_array(features.into_dyn())?;
        features_tensor.into_dyn()
      }
      None => {
        let empty_features = Tensor::from_array(
          Array2::<f32>::zeros((0, 3072)).into_dyn(),
        )?;
        empty_features.into_dyn()
      }
    };

    // Run the embedding model
    let inputs = vec![
      ("input_ids", input_ids_value),
      ("image_features", image_features_value),
      ("audio_features", audio_features_value),
    ];

    let outputs = self.embedding_session.run(inputs)?;

    // Get the inputs_embeds from the outputs
    let inputs_embeds =
      outputs[0].try_extract_tensor::<f32>()?.view().to_owned();

    // Convert from dynamic shape to fixed shape
    let shape = inputs_embeds.shape();
    let inputs_embeds_2d = Array2::from_shape_vec(
      (shape[0], shape[1]),
      inputs_embeds.iter().cloned().collect(),
    )?;

    Ok(inputs_embeds_2d)
  }

  /// Run the text model (language model) for generation
  fn run_text_model(
    &self,
    inputs_embeds: Array2<f32>,
    attention_mask: Option<Array2<i64>>,
    input_mode: InputMode,
  ) -> Result<Array2<f32>> {
    // Prepare attention mask if provided
    let attention_mask_value = match attention_mask {
      Some(mask) => {
        // Convert i64 to f32
        let mask_f32 = mask.mapv(|x| x as f32);
        Value::from_array(mask_f32.into_dyn())?
      }
      None => {
        // Create a default attention mask (all ones) with f32 type
        let seq_len = inputs_embeds.shape()[1];
        let mask = Array2::<f32>::ones((1, seq_len));
        Value::from_array(mask.into_dyn())?
      }
    };

    // Create input mode tensor
    let input_mode_value = Value::from_array(
      Array::from_elem((), input_mode as i32 as f32).into_dyn(),
    )?;

    // Run the text model
    let inputs = vec![
      (
        "inputs_embeds",
        Value::from_array(inputs_embeds.into_dyn())?,
      ),
      ("attention_mask", attention_mask_value),
      ("input_mode", input_mode_value),
    ];

    let outputs = self.text_session.run(inputs)?;

    // Get the logits from the outputs
    let logits =
      outputs[0].try_extract_tensor::<f32>()?.view().to_owned();

    // Convert from dynamic shape to fixed shape
    let shape = logits.shape();
    let logits_2d = Array2::from_shape_vec(
      (shape[0], shape[1]),
      logits.iter().cloned().collect(),
    )?;

    Ok(logits_2d)
  }

  /// Process an image and convert it to embeddings
  fn get_image_embeddings(
    &self,
    image: &[u8],
  ) -> Result<Array2<f32>> {
    // Process the image
    let (pixel_values, attention_mask, image_sizes) =
      self.process_image(image)?;

    // Run the vision model
    let inputs = vec![
      ("pixel_values", pixel_values),
      ("attention_mask", attention_mask),
      ("image_sizes", image_sizes),
    ];

    let outputs = self.vision_session.run(inputs)?;

    // Get the image features from the outputs
    let image_features =
      outputs[0].try_extract_tensor::<f32>()?.view().to_owned();

    // Convert from dynamic shape to fixed shape
    let shape = image_features.shape();
    let image_features_2d = Array2::from_shape_vec(
      (shape[0], shape[1]),
      image_features.iter().cloned().collect(),
    )?;

    Ok(image_features_2d)
  }

  /// Process an audio clip and convert it to embeddings
  fn get_audio_embeddings(
    &self,
    audio_data: &[f32],
    sample_rate: i32,
  ) -> Result<Array2<f32>> {
    // Process the audio
    let (
      audio_embeds,
      audio_attention_mask,
      audio_embed_sizes,
      input_mode,
    ) = self.process_audio(audio_data, sample_rate)?;

    // Run the speech model
    let inputs = vec![
      ("audio_embeds", audio_embeds),
      ("attention_mask", audio_attention_mask),
      ("audio_sizes", audio_embed_sizes),
      ("audio_projection_mode", input_mode),
    ];

    let outputs = self.speech_session.run(inputs)?;

    // Get the audio features from the outputs
    let audio_features =
      outputs[0].try_extract_tensor::<f32>()?.view().to_owned();

    // Convert from dynamic shape to fixed shape
    let shape = audio_features.shape();
    let audio_features_2d = Array2::from_shape_vec(
      (shape[0], shape[1]),
      audio_features.iter().cloned().collect(),
    )?;

    Ok(audio_features_2d)
  }

  /// Complete interface for processing text with optional image and audio
  pub fn process(
    &self,
    text: &str,
    image: Option<&[u8]>,
    audio: Option<(&[f32], i32)>,
  ) -> Result<String> {
    // Process text input - now returns a 2D array
    let input_ids = self.process_text(text)?;

    // Process image input if provided
    let image_features = match image {
      Some(img_data) => Some(self.get_image_embeddings(img_data)?),
      None => None,
    };

    // Process audio input if provided
    let audio_features = match audio {
      Some((audio_data, sample_rate)) => {
        Some(self.get_audio_embeddings(audio_data, sample_rate)?)
      }
      None => None,
    };

    // Determine input mode
    let input_mode =
      match (image_features.is_some(), audio_features.is_some()) {
        (true, true) => InputMode::VisionSpeech,
        (true, false) => InputMode::Vision,
        (false, true) => InputMode::Speech,
        (false, false) => InputMode::Language,
      };

    // Run the embedding model
    let inputs_embeds = self.run_embedding_model(
      input_ids,
      image_features,
      audio_features,
    )?;

    // Create attention mask (a 2D tensor)
    let seq_len = inputs_embeds.shape()[1];
    let attention_mask = Array2::<i64>::ones((1, seq_len));

    // Run the text model for generation
    let logits = self.run_text_model(
      inputs_embeds,
      Some(attention_mask),
      input_mode,
    )?;

    // Find the token with the highest probability
    let last_logits = logits.slice(s![0, ..]);
    let mut highest_idx = 0;
    let mut highest_value = f32::NEG_INFINITY;

    for (idx, &val) in last_logits.iter().enumerate() {
      if val > highest_value {
        highest_value = val;
        highest_idx = idx;
      }
    }

    // Convert token ID back to text
    let result = self
      .tokenizer
      .decode(&[highest_idx as u32], true)
      .map_err(|e| anyhow!("Failed to decode: {}", e))?;

    Ok(result)
  }
}
