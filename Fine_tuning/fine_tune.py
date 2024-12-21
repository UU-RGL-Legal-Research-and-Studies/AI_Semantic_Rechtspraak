import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import logging
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Function to clean text data
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text

# Load the data
file_path = 'Fine_tuning/qawaarderingsmethode_v2.xlsx'  # Update with your file path
data = pd.read_excel(file_path)

# Preprocess the data
data['query'] = data['query'].apply(clean_text)
data['answer'] = data['answer'].apply(clean_text)

# Check for missing values and drop them
data.dropna(inplace=True)

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Function to convert DataFrame to InputExamples
def df_to_input_examples(df):
    examples = []
    for index, row in df.iterrows():
        examples.append(InputExample(texts=[row['query'], row['answer']], label=1))
    return examples

# Convert cleaned DataFrame to InputExamples
input_examples = df_to_input_examples(data)

# Split the data into train and validation sets
train_examples, val_examples = train_test_split(input_examples, test_size=0.1, random_state=42)

# Verify the split
print(f"Number of training examples: {len(train_examples)}")
print(f"Number of validation examples: {len(val_examples)}")

# Load the pre-trained model
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

# Create DataLoader for training and validation
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=16)

# Define the loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

# Use an EarlyStopping callback
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='val-eval')

# Check if the output path exists, if not, create it
output_path = 'output/fine-tuned-model-waarderingsmethode-v2'
os.makedirs(output_path, exist_ok=True)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=5,  # Start with 5 epochs
    evaluation_steps=500,  # Evaluate every 500 steps for faster feedback
    warmup_steps=100,
    output_path=output_path,  # Correct the output path for the model
    checkpoint_path=os.path.join(output_path, 'checkpoints'),  # Add checkpoint path
    checkpoint_save_steps=1000,  # Save checkpoints every 1000 steps
    checkpoint_save_total_limit=2  # Limit to 2 checkpoints
)

# Save the trained model
model.save(output_path)
