#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark script to compare original vs optimized implementations
"""
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from models.transformer import TransformerEncoderDecoder as OriginalTransformer
from models.optimized_transformer import OptimizedTransformer
from models.attention import MultiHeadAttention


class PerformanceBenchmark:
    """Comprehensive benchmark for model performance"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.results = {}
    
    def benchmark_memory_usage(self, model, input_tensor, num_runs=10):
        """Benchmark memory usage"""
        torch.cuda.empty_cache()
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor)
        
        # Measure memory
        torch.cuda.reset_peak_memory_stats()
        
        start_memory = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated()
        
        return {
            'peak_memory_mb': (peak_memory - start_memory) / 1024 / 1024,
            'memory_per_run_mb': (peak_memory - start_memory) / 1024 / 1024 / num_runs
        }
    
    def benchmark_throughput(self, model, input_tensor, num_runs=100):
        """Benchmark throughput (samples/sec)"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = (input_tensor.size(0) * num_runs) / total_time
        
        return {
            'throughput_samples_sec': throughput,
            'latency_ms': (total_time / num_runs) * 1000
        }
    
    def benchmark_attention_efficiency(self, d_model=512, n_heads=8, seq_len=512, batch_size=32):
        """Compare attention implementations"""
        
        # Create test data
        x = torch.randn(batch_size, seq_len, d_model, device=self.device)
        mask = torch.ones(batch_size, seq_len, device=self.device)
        
        # Original attention (simulated)
        class OriginalAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_k = d_model // n_heads
                
                self.toqueries = nn.Linear(d_model, d_model)
                self.tokeys = nn.Linear(d_model, d_model)
                self.tovalues = nn.Linear(d_model, d_model)
                self.unifyheads = nn.Linear(d_model, d_model)
            
            def forward(self, x, mask=None):
                bs, qlen, dim = x.size()
                heads = self.n_heads
                
                query = self.toqueries(x).view(bs, qlen, heads, self.d_k).transpose(1, 2)
                key = self.tokeys(x).view(bs, qlen, heads, self.d_k).transpose(1, 2)
                value = self.tovalues(x).view(bs, qlen, heads, self.d_k).transpose(1, 2)
                
                query = query / (self.d_k ** (1 / 4))
                key = key / (self.d_k ** (1 / 4))
                
                scores = torch.matmul(query, key.transpose(-2, -1))
                if mask is not None:
                    mask = mask.unsqueeze(1)
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                dot = F.softmax(scores, dim=-1)
                out = torch.matmul(dot, value)
                out = out.transpose(1, 2).contiguous().view(bs, qlen, self.d_model)
                return self.unifyheads(out)
        
        # Optimized attention
        optimized_attention = MultiHeadAttention(d_model, n_heads).to(self.device)
        original_attention = OriginalAttention(d_model, n_heads).to(self.device)
        
        # Benchmark
        results = {}
        
        for name, attention in [('original', original_attention), ('optimized', optimized_attention)]:
            # Memory benchmark
            memory_results = self.benchmark_memory_usage(attention, x)
            
            # Throughput benchmark
            throughput_results = self.benchmark_throughput(attention, x)
            
            results[name] = {**memory_results, **throughput_results}
        
        return results
    
    def run_full_benchmark(self):
        """Run comprehensive benchmark"""
        
        configs = [
            {'d_model': 256, 'n_heads': 4, 'num_layers': 3, 'seq_len': 128, 'batch_size': 32},
            {'d_model': 512, 'n_heads': 8, 'num_layers': 6, 'seq_len': 256, 'batch_size': 16},
            {'d_model': 768, 'n_heads': 12, 'num_layers': 8, 'seq_len': 512, 'batch_size': 8},
        ]
        
        all_results = []
        
        for config in configs:
            print(f"Benchmarking config: {config}")
            
            # Create models
            original_model = OriginalTransformer(
                k=config['d_model'],
                heads=config['n_heads'],
                depth=config['num_layers'],
                num_emb=32000,
                num_emb_target=32000,
                max_len=config['seq_len']
            ).to(self.device)
            
            optimized_model = OptimizedTransformer(
                src_vocab_size=32000,
                tgt_vocab_size=32000,
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                num_encoder_layers=config['num_layers'],
                num_decoder_layers=config['num_layers'],
                max_seq_len=config['seq_len']
            ).to(self.device)
            
            # Create test data
            src = torch.randint(0, 32000, (config['batch_size'], config['seq_len']), device=self.device)
            tgt = torch.randint(0, 32000, (config['batch_size'], config['seq_len']), device=self.device)
            
            # Benchmark each model
            for model_name, model in [('original', original_model), ('optimized', optimized_model)]:
                print(f"  Testing {model_name}...")
                
                # Memory benchmark
                memory_results = self.benchmark_memory_usage(model, (src, tgt))
                
                # Throughput benchmark
                def model_forward(inputs):
                    src, tgt = inputs
                    return model(src, tgt)
                
                throughput_results = self.benchmark_throughput(model, (src, tgt))
                
                result = {
                    'config': config,
                    'model': model_name,
                    **memory_results,
                    **throughput_results
                }
                all_results.append(result)
        
        return pd.DataFrame(all_results)
    
    def create_visualization(self, results_df):
        """Create benchmark visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Memory usage comparison
        pivot_memory = results_df.pivot_table(
            values='peak_memory_mb', 
            index=['d_model', 'num_layers'], 
            columns='model'
        )
        
        pivot_memory.plot(kind='bar', ax=ax1)
        ax1.set_title('Memory Usage Comparison')
        ax1.set_ylabel('Memory (MB)')
        ax1.legend(['Original', 'Optimized'])
        
        # Throughput comparison
        pivot_throughput = results_df.pivot_table(
            values='throughput_samples_sec', 
            index=['d_model', 'num_layers'], 
            columns='model'
        )
        
        pivot_throughput.plot(kind='bar', ax=ax2)
        ax2.set_title('Throughput Comparison')
        ax2.set_ylabel('Samples/Second')
        ax2.legend(['Original', 'Optimized'])
        
        # Speedup ratio
        results_df['speedup'] = results_df.groupby(['d_model', 'num_layers']).apply(
            lambda x: x[x['model'] == 'optimized']['throughput_samples_sec'].iloc[0] /
                     x[x['model'] == 'original']['throughput_samples_sec'].iloc[0]
        ).values
        
        speedup_df = results_df[results_df['model'] == 'optimized'].copy()
        ax3.bar(range(len(speedup_df)), speedup_df['speedup'])
        ax3.set_title('Speedup Ratio')
        ax3.set_ylabel('Speedup (x)')
        ax3.set_xticks(range(len(speedup_df)))
        ax3.set_xticklabels([f"{row['d_model']}-{row['num_layers']}" for _, row in speedup_df.iterrows()])
        
        # Memory reduction
        results_df['memory_reduction'] = results_df.groupby(['d_model', 'num_layers']).apply(
            lambda x: (x[x['model'] == 'original']['peak_memory_mb'].iloc[0] -
                      x[x['model'] == 'optimized']['peak_memory_mb'].iloc[0]) /
                      x[x['model'] == 'original']['peak_memory_mb'].iloc[0] * 100
        ).values
        
        memory_df = results_df[results_df['model'] == 'optimized'].copy()
        ax4.bar(range(len(memory_df)), memory_df['memory_reduction'])
        ax4.set_title('Memory Reduction')
        ax4.set_ylabel('Reduction (%)')
        ax4.set_xticks(range(len(memory_df)))
        ax4.set_xticklabels([f"{row['d_model']}-{row['num_layers']}" for _, row in memory_df.iterrows()])
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run benchmark"""
    print("Starting performance benchmark...")
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark(device=device)
    
    # Run attention efficiency benchmark
    print("\n=== Attention Efficiency Benchmark ===")
    attention_results = benchmark.benchmark_attention_efficiency()
    for name, results in attention_results.items():
        print(f"{name}:")
        for key, value in results.items():
            print(f"  {key}: {value:.2f}")
    
    # Run full benchmark
    print("\n=== Full Model Benchmark ===")
    results_df = benchmark.run_full_benchmark()
    
    # Save results
    results_df.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to benchmark_results.csv")
    
    # Create visualization
    benchmark.create_visualization(results_df)
    
    # Summary
    print("\n=== Summary ===")
    avg_speedup = results_df[results_df['model'] == 'optimized']['throughput_samples_sec'].values / \
                  results_df[results_df['model'] == 'original']['throughput_samples_sec'].values
    avg_memory_reduction = (results_df[results_df['model'] == 'original']['peak_memory_mb'].values -
                           results_df[results_df['model'] == 'optimized']['peak_memory_mb'].values) / \
                           results_df[results_df['model'] == 'original']['peak_memory_mb'].values * 100
    
    print(f"Average speedup: {np.mean(avg_speedup):.2f}x")
    print(f"Average memory reduction: {np.mean(avg_memory_reduction):.1f}%")


if __name__ == "__main__":
    main()