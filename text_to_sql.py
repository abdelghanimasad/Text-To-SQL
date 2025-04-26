import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import random
import numpy as np

# -------------------- Config --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 3e-4
TEACHER_FORCING_RATIO = 0.5
CLIP_GRAD = 1.0

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

# -------------------- Load Data --------------------
print("Loading data...")
df = pd.read_csv("processed_text_to_sql_data.csv")
df.drop(columns=["schema"], inplace=True, errors='ignore')  # Drop schema column if it exists
df.rename(columns={"question": "input_text", "query": "target_text"}, inplace=True)

# Keep only rows where both columns are not empty and not null
df = df.dropna(subset=['input_text', 'target_text'])
df = df[df['input_text'].str.strip() != '']
df = df[df['target_text'].str.strip() != '']

print(f"Dataset size after cleaning: {len(df)}")

# Optional: Take a subset for faster experimentation/debugging
# df = df.sample(10000, random_state=42)

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

# -------------------- Tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained("t5-small")
vocab_size = tokenizer.vocab_size
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id
sos_token_id = tokenizer.pad_token_id  # Using pad token as start token since T5 uses pad for decoder start

# -------------------- Dataset Class --------------------
class SQLDataset(Dataset):
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q = str(self.data.iloc[idx]["input_text"])
        a = str(self.data.iloc[idx]["target_text"])

        # Tokenize inputs (questions)
        enc = tokenizer(q, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        input_ids = enc.input_ids.squeeze(0)
        input_mask = enc.attention_mask.squeeze(0)
        
        # Tokenize targets (SQL queries)
        dec = tokenizer(a, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        target_ids = dec.input_ids.squeeze(0)
        
        # Create shifted decoder input (right-shifted with start token)
        # For the decoder input, we prepend the pad token (as BOS)
        dec_input = torch.cat([
            torch.tensor([pad_token_id]), 
            target_ids[:-1]
        ])
        
        return {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "decoder_input_ids": dec_input,
            "target_ids": target_ids
        }

print("Preparing datasets and dataloaders...")
train_ds = SQLDataset(train_df.reset_index(drop=True))
val_ds = SQLDataset(val_df.reset_index(drop=True))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# -------------------- Seq2Seq with Attention --------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, 
                          dropout=dropout if n_layers > 1 else 0,
                          bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_lengths=None):
        # src: [batch_size, src_len]
        
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, embedding_dim]
        
        if src_lengths is not None:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_outputs, hidden = self.rnn(packed_embedded)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)
        
        # outputs: [batch_size, src_len, hidden_dim * 2]
        # hidden: [n_layers * 2, batch_size, hidden_dim]
        
        # Combine bidirectional states for the hidden state
        # First, separate forward and backward hidden states
        hidden = hidden.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)
        # hidden: [n_layers, 2, batch_size, hidden_dim]
        
        # Concatenate forward and backward states
        # and transform to match decoder dimension
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        # hidden: [n_layers, batch_size, hidden_dim * 2]
        
        # Project to decoder hidden dimension
        hidden = self.fc(hidden)
        # hidden: [n_layers, batch_size, hidden_dim]
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        
        # Encoder hidden is bidirectional, so we need to account for doubled size
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, dec_hidden_dim]
        # encoder_outputs: [batch_size, src_len, enc_hidden_dim * 2]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat hidden across source sequence length
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, dec_hidden_dim]
        
        # Calculate energy using attention network
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, dec_hidden_dim]
        
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        
        # Apply mask if provided (1 for tokens, 0 for padding)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=1)
        # attention_weights: [batch_size, src_len]
        
        # Use attention weights to create context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # context: [batch_size, 1, enc_hidden_dim * 2]
        context = context.squeeze(1)
        # context: [batch_size, enc_hidden_dim * 2]
        
        return context, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_hidden_dim, dec_hidden_dim, attention, dropout=0.1):
        super().__init__()
        
        self.output_dim = vocab_size
        self.attention = attention
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Input to GRU will be embedding + context vector
        self.rnn = nn.GRU(embedding_dim + (enc_hidden_dim * 2), dec_hidden_dim, batch_first=True)
        
        # Output layer
        self.fc_out = nn.Linear(dec_hidden_dim + (enc_hidden_dim * 2) + embedding_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask=None):
        # input: [batch_size] (single token)
        # hidden: [1, batch_size, dec_hidden_dim]
        # encoder_outputs: [batch_size, src_len, enc_hidden_dim * 2]
        
        input = input.unsqueeze(1)  # input: [batch_size, 1]
        
        # Embed the input token
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, embedding_dim]
        
        # Calculate attention and get context vector
        # Get current hidden state for attention
        # Extract the last layer's hidden state
        hidden_for_attn = hidden.squeeze(0)
        # hidden_for_attn: [batch_size, dec_hidden_dim]
        
        context, attention = self.attention(hidden_for_attn, encoder_outputs, mask)
        # context: [batch_size, enc_hidden_dim * 2]
        
        # Expand context to sequence length of 1
        context = context.unsqueeze(1)
        # context: [batch_size, 1, enc_hidden_dim * 2]
        
        # Combine embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input: [batch_size, 1, embedding_dim + enc_hidden_dim * 2]
        
        # Pass through GRU
        output, hidden = self.rnn(rnn_input, hidden)
        # output: [batch_size, 1, dec_hidden_dim]
        # hidden: [1, batch_size, dec_hidden_dim]
        
        # Remove sequence dimension
        output = output.squeeze(1)
        # output: [batch_size, dec_hidden_dim]
        
        context = context.squeeze(1)
        # context: [batch_size, enc_hidden_dim * 2]
        
        embedded = embedded.squeeze(1)
        # embedded: [batch_size, embedding_dim]
        
        # Final prediction layer combining all info
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        # prediction: [batch_size, vocab_size]
        
        return prediction, hidden

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, src_mask, tgt, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # src_mask: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim
        
        # Prepare tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        
        # Get encoder outputs and hidden state
        encoder_outputs, hidden = self.encoder(src)
        
        # The encoder hidden state needs to be reshaped for the decoder
        # We only use the last layer's hidden state to initialize decoder
        hidden = hidden[-1:, :, :]  # [1, batch_size, hidden_dim]
        
        # First input to the decoder is the <sos> token (we use pad token for this)
        input = tgt[:, 0]
        
        # Track attentions for visualization (optional)
        attentions = []
        
        for t in range(1, tgt_len):
            # Get decoder output for this step
            output, hidden = self.decoder(input, hidden, encoder_outputs, src_mask)
            
            # Store output
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Use either teacher forcing or model's prediction
            input = tgt[:, t] if teacher_force else top1
            
        return outputs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize the model components
print("Building model...")
enc = Encoder(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, n_layers=2, dropout=0.1)
attention = Attention(HIDDEN_DIM, HIDDEN_DIM)
dec = Decoder(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM, attention, dropout=0.1)
model = Seq2SeqWithAttention(enc, dec).to(DEVICE)

print(f"The model has {count_parameters(model):,} trainable parameters")

# -------------------- Training Setup --------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

# -------------------- Training Loop --------------------
def train_epoch(model, train_loader, optimizer, criterion, clip=CLIP_GRAD, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(train_loader):
        # Get batch data
        src = batch["input_ids"].to(DEVICE)
        src_mask = batch["attention_mask"].to(DEVICE)
        tgt = batch["decoder_input_ids"].to(DEVICE)
        tgt_ids = batch["target_ids"].to(DEVICE) 
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, src_mask, tgt, teacher_forcing_ratio)
        
        # Reshape output and target for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)  # Skip first position (start token)
        tgt_ids = tgt_ids[:, 1:].contiguous().view(-1)  # Skip first position
        
        # Calculate loss
        loss = criterion(output, tgt_ids)
        
        # Backward pass and update
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    return epoch_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Get batch data
            src = batch["input_ids"].to(DEVICE)
            src_mask = batch["attention_mask"].to(DEVICE)
            tgt = batch["decoder_input_ids"].to(DEVICE)
            tgt_ids = batch["target_ids"].to(DEVICE)
            
            # Forward pass
            output = model(src, src_mask, tgt, teacher_forcing_ratio=0)  # No teacher forcing during evaluation
            
            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            tgt_ids = tgt_ids[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, tgt_ids)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(val_loader)

# -------------------- Training Loop --------------------
best_valid_loss = float('inf')

def generate_sql(model, question, max_len=MAX_LEN):
    model.eval()
    
    # Tokenize the input question
    enc = tokenizer(question, padding="max_length", truncation=True, 
                   max_length=MAX_LEN, return_tensors="pt")
    src = enc.input_ids.to(DEVICE)
    src_mask = enc.attention_mask.to(DEVICE)
    
    # Get encoder outputs and hidden state
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        
        # The encoder hidden state needs to be reshaped for the decoder
        # We only use the last layer's hidden state to initialize decoder
        hidden = hidden[-1:, :, :]  # [1, batch_size, hidden_dim]
        
        # Start with <sos> token (we use pad token as start token)
        input = torch.tensor([pad_token_id]).to(DEVICE)
        
        generated_tokens = []
        
        for _ in range(max_len):
            # Get decoder output
            output, hidden = model.decoder(input, hidden, encoder_outputs, src_mask)
            
            # Get the highest predicted token
            pred_token = output.argmax(1)
            
            # If we predict <eos> token, stop generation
            if pred_token.item() == eos_token_id:
                break
                
            generated_tokens.append(pred_token.item())
            
            # Next input is the predicted token
            input = pred_token
    
    # Decode the generated tokens
    sql_query = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return sql_query

# -------------------- Main Training Loop --------------------
print("Starting training...")
try:
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        
        # Evaluate
        valid_loss = evaluate(model, val_loader, criterion)
        
        # Print epoch stats
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}")
        
        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"seq2seq_attn_best_model.pt")
            print(f"✅ Best model saved (val loss: {valid_loss:.4f})")
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
        }, f"seq2seq_attn_epoch_{epoch}.pt")
        print(f"✅ Checkpoint saved for epoch {epoch}")

        # Test model with sample questions
        if epoch % 2 == 0:
            print("\nTesting with sample questions:")
            test_questions = [
                "List all students older than 20",
                "Get the names of students with GPA greater than 3.5",
                "SELECT all employees who joined after 2019",
                "Find total sales for each product category"
            ]
            
            for q in test_questions:
                sql = generate_sql(model, q)
                print(f"\nQuestion: {q}")
                print(f"Generated SQL: {sql}")

except KeyboardInterrupt:
    print("Training interrupted!")

# -------------------- Test Function --------------------
def test_model(model_path, question):
    # Load the model
    enc = Encoder(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, n_layers=2, dropout=0.1)
    attention = Attention(HIDDEN_DIM, HIDDEN_DIM)
    dec = Decoder(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM, attention, dropout=0.1)
    model = Seq2SeqWithAttention(enc, dec).to(DEVICE)
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Generate SQL
    sql = generate_sql(model, question)
    return sql

# Example usage
if __name__ == "__main__":
    # Test with a custom question
    question = "Get all customers who spent more than $1000 last month"
    model_path = "seq2seq_attn_best_model.pt"  # Change to the best model path
    sql = test_model(model_path, question)
    print(f"\nQuestion: {question}")
    print(f"Generated SQL: {sql}")