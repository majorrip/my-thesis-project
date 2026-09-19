import torch
from model import VisionJEPA, compute_stable_local_loss

def run_training_experiment(epochs=10, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Training Pipeline on: {device}")
    
    model = VisionJEPA().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Synthetic batch representing full images, masked context, and action locations
    full_images = torch.rand(batch_size, 3, 64, 64).to(device)
    partial_images = full_images.clone()
    partial_images[:, :, 16:48, 16:48] = 0.0 # Crop central patch
    spatial_actions = torch.tensor([[0.25, 0.25, 0.5, 0.5]] * batch_size, dtype=torch.float32).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        s_pred, s_tgt, s_t = model(partial_images, full_images, spatial_actions)
        
        loss = compute_stable_local_loss(s_pred, s_tgt.detach(), s_t, variance_threshold=1.0)
        loss.backward()
        optimizer.step()
        model.update_target_encoder()
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_training_experiment()