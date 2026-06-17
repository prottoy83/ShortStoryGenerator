import torch

def train_model(epochs, dataset, model, tokenizer, device, checkpoint):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0 

        for batches, (X_b, y_b) in enumerate(dataset):
            X_b = X_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()


            pred = model(X_b)
            vocab_size = pred.size(-1)

            loss = criterion(pred.view(-1, vocab_size), y_b.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if batches % 500 == 0:
                print(f"Epoch {epoch+1} | Batch {batches} | Current Loss: {loss.item():.4f}")

                checkpoint_dict = {
                    'model_state_dict': model.state_dict(),
                    'w2i': tokenizer.w2i,
                    'i2w': tokenizer.i2w,
                    'vocab_size': len(tokenizer),
                    'embed_size': model.embedding.embedding_dim,
                    'hidden_size': model.lstm.hidden_size
                }

                torch.save(checkpoint_dict, checkpoint)
                print(f"Checkpoint Created")
        
        avg_loss = total_loss / (batches+1)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    print("Training Complete")
    
    