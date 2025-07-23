import os
import pandas as pd
from datasets import load_dataset
import torchaudio
from tqdm import tqdm


def convert_hf_dataset_to_csv_and_wav(dataset_name, output_csv_path, output_audio_dir):
    """
    Downloads a Hugging Face dataset, saves audio data to .wav files,
    and creates a CSV metadata file with paths to the audio.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub.
        output_csv_path (str): The path to save the output CSV file.
        output_audio_dir (str): The directory to save the .wav audio files.
    """
    print(f"Loading dataset '{dataset_name}' from Hugging Face...")
    # Load the dataset from the Hugging Face Hub.
    # The 'streaming=True' option is good practice for large datasets,
    # but for this size, we can load it directly. Let's load it all.
    try:
        dataset = load_dataset(dataset_name, split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have an internet connection and the 'datasets' library is installed.")
        return

    print(f"Dataset loaded successfully. Total rows: {len(dataset)}")

    # Create the output directory for audio files if it doesn't exist.
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir)
        print(f"Created audio directory: '{output_audio_dir}'")

    # This list will hold the metadata for each row, which will become our CSV.
    metadata_records = []

    print("Processing dataset rows, saving audio, and preparing metadata...")
    # Use tqdm for a nice progress bar
    for i, item in tqdm(enumerate(dataset), desc="Processing files"):
        # --- 1. Extract Audio Data ---
        audio_data = item.get('audio').get_all_samples()
        audio = audio_data.data
        sr = audio_data.sample_rate

        # --- 2. Construct the Filename ---
        category = item.get('category', 'unknown').replace(" ", "_") # Sanitize category name
        language = item.get('language', 'unknown')
        evolution_depth = item.get('evolution_depth', 'unknown')

        filename = f"{str(i).zfill(4)}_{category}_{language}_{evolution_depth}.wav"
        relative_audio_path = os.path.join(output_audio_dir, filename)

        # --- 3. Save the Audio File ---
        try:
            torchaudio.save(relative_audio_path, audio, sr)
        except Exception as e:
            print(f"Error saving audio file {relative_audio_path}: {e}")
            continue

        # --- 4. Prepare the Metadata Record ---
        # Create a copy of the item's data and remove the original audio column.
        record = {key: value for key, value in item.items() if key != 'audio'}
        # Add the new column for the relative audio path.
        record['audio_path'] = relative_audio_path
        metadata_records.append(record)

    if not metadata_records:
        print("No records were processed. Exiting.")
        return

    # --- 5. Create and Save the CSV File ---
    print("Converting metadata to a pandas DataFrame...")
    df = pd.DataFrame(metadata_records)

    # Reorder columns to have audio_path at the end, if desired
    if 'audio_path' in df.columns:
        cols = [col for col in df.columns if col != 'audio_path'] + ['audio_path']
        df = df[cols]

    print(f"Saving metadata to '{output_csv_path}' with '|' separator...")
    try:
        df.to_csv(output_csv_path, sep='|', index=False)
        print("Conversion complete!")
        print(f"✅ Metadata saved to: {output_csv_path}")
        print(f"✅ Audio files saved in: {output_audio_dir}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    DATASET_NAME = "bosonai/EmergentTTS-Eval"
    OUTPUT_CSV = "metadata.csv"
    OUTPUT_AUDIO = "audio"

    # --- Run the Conversion ---
    convert_hf_dataset_to_csv_and_wav(DATASET_NAME, OUTPUT_CSV, OUTPUT_AUDIO)

