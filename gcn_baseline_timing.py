#!/usr/bin/env python3
"""
GCN baseline timing: CPU vs GPU (FIX FINAL)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import time

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def benchmark_dataset(dataset_name):
    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = Planetoid(root='/tmp/Planetoid', name=dataset_name)
    data = dataset[0]
    num_features, num_classes = dataset.num_features, dataset.num_classes
    
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name} (N={data.num_nodes}, F={num_features}, C={num_classes})")
    print(f"{'='*50}")
    
    # CPU
    model_cpu = GCN(num_features, num_classes).to(device_cpu)
    optimizer_cpu = torch.optim.Adam(model_cpu.parameters(), lr=0.01, weight_decay=5e-4)
    
    data_cpu = data.to(device_cpu)
    start_cpu = time.time()
    model_cpu.train()
    optimizer_cpu.zero_grad()
    out_cpu = model_cpu(data_cpu)
    loss_cpu = F.nll_loss(out_cpu[data.train_mask], data.y[data.train_mask])
    loss_cpu.backward()
    optimizer_cpu.step()
    time_cpu = time.time() - start_cpu
    
    acc_cpu = (out_cpu.argmax(1)[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    
    # GPU 
    if torch.cuda.is_available():
        model_gpu = GCN(num_features, num_classes).to(device_gpu)
        optimizer_gpu = torch.optim.Adam(model_gpu.parameters(), lr=0.01, weight_decay=5e-4)
        
        data_gpu = data.to(device_gpu)
        start_gpu = time.time()
        model_gpu.train()
        optimizer_gpu.zero_grad()
        out_gpu = model_gpu(data_gpu)
        loss_gpu = F.nll_loss(out_gpu[data.train_mask], data.y[data.train_mask].to(device_gpu))
        loss_gpu.backward()
        optimizer_gpu.step()
        time_gpu = time.time() - start_gpu
        
        # FIX: Tudo em CPU
        acc_gpu = (out_gpu.argmax(1)[data_gpu.test_mask].cpu() == data.y[data_gpu.test_mask].cpu()).float().mean().item()
        speedup = time_cpu / time_gpu
        
        print(f"CPU:    {time_cpu*1000:.1f}ms | Loss: {loss_cpu:.4f} | Test ACC: {acc_cpu:.1%}")
        print(f"GPU:    {time_gpu*1000:.1f}ms | Loss: {loss_gpu:.4f} | Test ACC: {acc_gpu:.1%}")
        print(f"Speedup: {speedup:.1f}x")
    else:
        print(f"CPU:    {time_cpu*1000:.1f}ms | Loss: {loss_cpu:.4f} | Test ACC: {acc_cpu:.1%}")
        print("GPU:    Não disponível")
    
    print("-" * 50)

if __name__ == "__main__":
    datasets = ['Cora', 'Citeseer', 'Pubmed']
    print("GCN Baseline Timing (CPU vs GPU) - 1 forward+backward\n")
    
    for ds in datasets:
        benchmark_dataset(ds)
    
    print("\n✅ Benchmark concluído!")
