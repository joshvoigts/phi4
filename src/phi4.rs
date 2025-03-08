use anyhow::{anyhow, Result};
use half::f16;
use ndarray::{s, Array, Array1, Array2, Array3, Array4, Axis};
use ort::environment::Environment;
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use ort::value::TensorValueType;
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

    let text_session = SessionBuilder::new()?
      .with_execution_providers([
        CUDAExecutionProvider::default().build()
      ])?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .commit_from_file(text_path)?;

    dbg!("Loaded models");

    // Load the tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)
      .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

    dbg!("Loaded tokenizer");

    Ok(Self {
      vision_session,
      speech_session,
      embedding_session,
      text_session,
      tokenizer,
      environment,
    })
  }

  /// Convert f32 array to f16 array
  fn convert_to_f16<D>(&self, arr: Array<f32, D>) -> Array<f16, D>
  where
    D: ndarray::Dimension,
  {
    arr.mapv(|x| f16::from_f32(x))
  }

  /// Convert f16 array to f32 array
  fn convert_to_f32<D>(&self, arr: Array<f16, D>) -> Array<f32, D>
  where
    D: ndarray::Dimension,
  {
    arr.mapv(|x| x.to_f32())
  }

  /// Process image input for the vision model
  fn process_image(
    &self,
    image: &[u8],
  ) -> Result<(Value, Value, Value)> {
    // Create image input tensor (batch_size, max_num_crops, channels, height, width)
    let batch_size = 1;
    let max_num_crops = 5; // Adjust based on your needs
    let channels = 3;
    let height = 448;
    let width = 448;

    // Create f32 arrays first
    let pixel_values_f32 = Array::<f32, _>::zeros((
      batch_size,
      max_num_crops,
      channels,
      height,
      width,
    ));

    // For the attention mask, using the correct name based on your session info
    let attention_mask_f32 =
      Array::<f32, _>::ones((batch_size, max_num_crops, 32, 32));

    // Convert to f16
    let pixel_values_f16 = self.convert_to_f16(pixel_values_f32);
    let attention_mask_f16 = self.convert_to_f16(attention_mask_f32);

    // Create image sizes - This remains i64
    let image_sizes = Array::<i64, _>::from_shape_vec(
      (batch_size, 2),
      vec![height as i64, width as i64],
    )?;

    // Create tensors with explicit type
    let pixel_values_tensor =
      Tensor::from_array(pixel_values_f16.into_dyn())?;
    let attention_mask_tensor =
      Tensor::from_array(attention_mask_f16.into_dyn())?;
    let image_sizes_tensor =
      Tensor::from_array(image_sizes.into_dyn())?;

    // Convert to Value
    let pixel_values_value = pixel_values_tensor.into_dyn();
    let attention_mask_value = attention_mask_tensor.into_dyn();
    let image_sizes_value = image_sizes_tensor.into_dyn();

    Ok((pixel_values_value, attention_mask_value, image_sizes_value))
  }

  /// Process audio input for the speech model
  fn process_audio(
    &self,
    audio_data: &[f32],
    sample_rate: i32,
  ) -> Result<(Value, Value, Value, Value)> {
    // Example code for creating placeholder tensors
    let batch_size = 1;
    let num_frames = 128; // Adjust based on your needs
    let feature_size = 80; // Mel spectrogram features

    // Create audio input tensor (batch_size, num_frames, feature_size) in f32
    let audio_embeds_f32 =
      Array::<f32, _>::zeros((batch_size, num_frames, feature_size));

    // Convert to f16
    let audio_embeds_f16 = self.convert_to_f16(audio_embeds_f32);

    // Create audio attention mask (using bool type based on session info)
    let audio_attention_mask =
      Array::<bool, _>::from_elem((batch_size, num_frames), true);

    // Create audio embed sizes (remains as i64)
    let audio_embed_sizes =
      Array::from_shape_vec((batch_size,), vec![num_frames as i64])?;

    // Create input mode - always Speech for this function
    let input_mode = Array::from_elem((1,), InputMode::Speech as i64);

    // Create tensors
    let audio_embeds_tensor =
      Tensor::from_array(audio_embeds_f16.into_dyn())?;
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
      audio_embed_sizes_value,
      input_mode_value,
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
    image_features: Option<Array2<f16>>,
    audio_features: Option<Array2<f16>>,
  ) -> Result<Array3<f16>> {
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
        // Create empty f16 features - with correct dimension 3072
        let empty_features = Tensor::from_array(
          Array2::<f16>::zeros((0, 3072)).into_dyn(),
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
        // Create empty f16 features
        let empty_features = Tensor::from_array(
          Array2::<f16>::zeros((0, 3072)).into_dyn(),
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

    dbg!("Starting embedding");

    let outputs = self.embedding_session.run(inputs)?;

    dbg!("Finished embedding");

    // Get the inputs_embeds from the outputs
    let inputs_embeds =
      outputs[0].try_extract_tensor::<f16>()?.view().to_owned();

    // Convert from dynamic shape to fixed shape
    let shape = inputs_embeds.shape();
    let inputs_embeds_3d = Array3::from_shape_vec(
      (shape[0], shape[1], shape[2]),
      inputs_embeds.iter().cloned().collect(),
    )?;

    Ok(inputs_embeds_3d)
  }

  /// Create empty past key/value tensors for the text model's initial call
  fn create_empty_past_key_values(
    &self,
    batch_size: usize,
  ) -> Result<Vec<(String, Value<TensorValueType<f16>>)>> {
    let mut past_kv_inputs = Vec::with_capacity(64); // 32 keys + 32 values

    for i in 0..32 {
      // Create empty key tensor [batch_size, 8, 0, 128]
      let past_key = Array4::<f16>::zeros((batch_size, 8, 0, 128));
      let past_key_tensor = Tensor::from_array(past_key.into_dyn())?;
      past_kv_inputs.push((
        format!("past_key_values.{}.key", i),
        past_key_tensor,
      ));

      // Create empty value tensor [batch_size, 8, 0, 128]
      let past_value = Array4::<f16>::zeros((batch_size, 8, 0, 128));
      let past_value_tensor =
        Tensor::from_array(past_value.into_dyn())?;
      past_kv_inputs.push((
        format!("past_key_values.{}.value", i),
        past_value_tensor,
      ));
    }

    Ok(past_kv_inputs)
  }

  /// Run the text model (language model) for generation
  fn run_text_model(
    &self,
    inputs_embeds: Array3<f16>,
    attention_mask: Option<Array2<i64>>,
    past_key_values: Option<
      Vec<(String, Value<TensorValueType<f16>>)>,
    >,
  ) -> Result<(Array2<f32>, Vec<(String, Value<TensorValueType<f16>>)>)>
  {
    let batch_size = inputs_embeds.shape()[0];
    let seq_len = inputs_embeds.shape()[1];

    // Prepare attention mask if provided and convert to f16
    let mask_f16 = match attention_mask {
      Some(mask) => {
        // Convert i64 mask to f16
        mask.mapv(|x| f16::from_f32(x as f32))
      }
      None => {
        // Create a default attention mask with f16 type
        Array2::<f16>::ones((batch_size, seq_len))
      }
    };
    let attention_mask_tensor =
      Tensor::from_array(mask_f16.clone().into_dyn())?;

    // Create position_ids tensor and convert to f16
    let position_ids_i64 = if let Some(_) = &past_key_values {
      // For subsequent runs with past kv, position_ids are just the current position
      let past_seq_len = mask_f16.shape()[1] - 1;
      Array2::<i64>::from_shape_fn((batch_size, 1), |(_, _)| {
        past_seq_len as i64
      })
    } else {
      // For first run, position_ids are 0 to seq_len-1
      Array2::<i64>::from_shape_fn((batch_size, seq_len), |(_, i)| {
        i as i64
      })
    };
    let position_ids =
      position_ids_i64.mapv(|x| f16::from_f32(x as f32));
    let position_ids_tensor =
      Tensor::from_array(position_ids.into_dyn())?;

    // Handle inputs_embeds - already a 3D array (batch_size, seq_len, hidden_size)
    let inputs_embeds_3d = if let Some(_) = &past_key_values {
      // For subsequent runs, we only need the last token
      inputs_embeds.slice(s![.., -1.., ..]).to_owned()
    } else {
      // For first run, use the full sequence
      inputs_embeds
    };

    // Create tensor for inputs
    let inputs_embeds_tensor =
      Tensor::from_array(inputs_embeds_3d.into_dyn())?;

    // Build all inputs
    let mut inputs = Vec::new();
    inputs.push(("inputs_embeds".to_string(), inputs_embeds_tensor));
    inputs
      .push(("attention_mask".to_string(), attention_mask_tensor));
    inputs.push(("position_ids".to_string(), position_ids_tensor));

    if let Some(pkv) = past_key_values {
      for (key, value) in pkv {
        inputs.push((key, value));
      }
    } else {
      let empty_kv = self
        .create_empty_past_key_values(batch_size)
        .unwrap_or_default();

      for (key, value) in empty_kv {
        inputs.push((key, value));
      }
    }

    // Run the text model
    let outputs = self.text_session.run(inputs)?;

    // Get the logits from the outputs
    let logits_f16 =
      outputs[0].try_extract_tensor::<f16>()?.view().to_owned();

    // Convert logits from f16 to f32
    let logits_f32 = self.convert_to_f32(logits_f16);

    // Convert from dynamic shape to fixed shape
    let shape = logits_f32.shape();
    let logits_2d = Array2::from_shape_vec(
      (shape[0], shape[1]),
      logits_f32.iter().cloned().collect(),
    )?;

    // Extract the present key/values to return for the next run
    let mut present_key_values = Vec::with_capacity(64);
    for i in 0..32 {
      let key_idx = i * 2 + 1; // Logits is at index 0, then present.0.key, present.0.value, ...
      let value_idx = i * 2 + 2;

      if key_idx < outputs.len() && value_idx < outputs.len() {
        // Extract the tensors and create new Values with specific type
        let key_tensor =
          outputs[key_idx].try_extract_tensor::<f16>()?;
        let key_value =
          Tensor::from_array(key_tensor.view().into_dyn())?;
        present_key_values
          .push((format!("past_key_values.{}.key", i), key_value));

        let value_tensor =
          outputs[value_idx].try_extract_tensor::<f16>()?;
        let value_value =
          Tensor::from_array(value_tensor.view().into_dyn())?;
        present_key_values.push((
          format!("past_key_values.{}.value", i),
          value_value,
        ));
      }
    }

    Ok((logits_2d, present_key_values))
  }

  /// Process an image and convert it to embeddings
  fn get_image_embeddings(
    &self,
    image: &[u8],
  ) -> Result<Array2<f16>> {
    // Process the image
    let (pixel_values, attention_mask, image_sizes) =
      self.process_image(image)?;

    // Run the vision model
    let inputs = vec![
      ("pixel_values", pixel_values),
      ("image_attention_mask", attention_mask), // Updated name based on session info
      ("image_sizes", image_sizes),
    ];

    let outputs = self.vision_session.run(inputs)?;

    // Get the image features from the outputs
    let image_features =
      outputs[0].try_extract_tensor::<f16>()?.view().to_owned();

    // Convert from dynamic shape to fixed shape
    let shape = image_features.shape();
    // Using dimension 3072 from the session info
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
  ) -> Result<Array2<f16>> {
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
      ("audio_attention_mask", audio_attention_mask),
      ("audio_sizes", audio_embed_sizes),
      ("audio_projection_mode", input_mode),
    ];

    let outputs = self.speech_session.run(inputs)?;

    // Get the audio features from the outputs as f16
    let audio_features =
      outputs[0].try_extract_tensor::<f16>()?.view().to_owned();

    // Convert from dynamic shape to fixed shape
    let shape = audio_features.shape();
    let audio_features_2d = Array2::from_shape_vec(
      (shape[0], shape[1]),
      audio_features.iter().cloned().collect(),
    )?;

    Ok(audio_features_2d)
  }

  /// Generate text one token at a time
  pub fn generate(
    &self,
    text: &str,
    image: Option<&[u8]>,
    audio: Option<(&[f32], i32)>,
    max_new_tokens: usize,
  ) -> Result<String> {
    dbg!("Processing text");
    // Process text input
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
    dbg!("Running embedding model");
    let inputs_embeds = self.run_embedding_model(
      input_ids.clone(),
      image_features,
      audio_features,
    )?;

    // Create attention mask for the initial sequence
    // Use the second dimension (sequence length) from the 3D tensor
    let mut seq_len = inputs_embeds.shape()[1];
    let mut attention_mask = Array2::<i64>::ones((1, seq_len));

    // Initial run of the text model without past key values
    dbg!("Running text model initial run");
    let (mut logits, mut past_key_values) = self.run_text_model(
      inputs_embeds,
      Some(attention_mask.clone()),
      None,
    )?;

    // Store all generated token IDs
    let mut all_token_ids = input_ids.to_owned();

    // Generate tokens one at a time
    for _ in 0..max_new_tokens {
      // Get the next token ID (from the last position)
      let next_token_logits = logits.slice(s![-1, ..]);

      let mut next_token_id = 0;
      let mut max_logit = f32::NEG_INFINITY;

      for (idx, &val) in next_token_logits.iter().enumerate() {
        if val > max_logit {
          max_logit = val;
          next_token_id = idx;
        }
      }

      // Add the new token ID to our collection
      let next_token_id_array =
        Array2::from_elem((1, 1), next_token_id as i64);

      // Create a new Array with the appended token
      let mut new_tokens = all_token_ids.to_owned();
      new_tokens.append(Axis(1), next_token_id_array.view())?;
      all_token_ids = new_tokens;

      // Check for EOS token (this would depend on your tokenizer configuration)
      if next_token_id == 1 {
        // Assuming 1 is EOS, adjust as needed
        break;
      }

      // Update attention mask for the next token
      seq_len += 1;
      attention_mask = Array2::<i64>::ones((1, seq_len));

      // Create embeddings for just the new token for the next iteration
      let next_token_embeds =
        self.run_embedding_model(next_token_id_array, None, None)?;

      // Run the text model for the next token with past key values
      let (new_logits, new_past_key_values) = self.run_text_model(
        next_token_embeds,
        Some(attention_mask.clone()),
        Some(past_key_values),
      )?;

      logits = new_logits;
      past_key_values = new_past_key_values;
    }

    // Decode all generated tokens
    let output_ids: Vec<u32> =
      all_token_ids.iter().map(|&id| id as u32).collect();

    let result = self
      .tokenizer
      .decode(&output_ids, true)
      .map_err(|e| anyhow!("Failed to decode: {}", e))?;

    Ok(result)
  }

  /// Simple interface for processing text with optional image and audio (no generation)
  pub fn process(
    &self,
    text: &str,
    image: Option<&[u8]>,
    audio: Option<(&[f32], i32)>,
  ) -> Result<String> {
    self.generate(text, image, audio, 1)
  }
}
