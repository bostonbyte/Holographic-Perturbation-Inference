import struct
import numpy as np

class HoloSerializer:
    MAGIC = b'HOLO'
    VERSION = 1
    
    @staticmethod
    def save(filepath, u_list, s_list, v_list, head_dim, rank, seq_len):
        """
        Saves the compressed KV cache (lists of U, S, V for each layer) to a binary file.
        Format:
            Header:
                Magic (4)
                Version (4)
                NumLayers (4)
                NumHeads (4)
                HeadDim (4)
                Rank (4)
                SeqLen (4)
            Payload:
                For each Layer:
                    For each Head:
                        U (HeadDim * Rank * 2 bytes)
                        S (Rank * 2 bytes)
                        V (Rank * SeqLen * 2 bytes)
        """
        num_layers = len(u_list)
        # u_list[0] shape is [NumHeads, HeadDim, Rank] or [NumHeads, Rank, HeadDim]? 
        # From experiment Phase 1: compress_matrix returns [NumHeads, SeqLen, HeadDim]...
        # Wait, Phase 1 was doing local operations.
        # Here we assume the inputs are the decomposed matrices U, S, V ready for storage.
        # U: [NumHeads, HeadDim, Rank]
        # S: [NumHeads, Rank]
        # V: [NumHeads, Rank, SeqLen]
        
        num_heads = u_list[0].shape[0]
        
        with open(filepath, 'wb') as f:
            # Header
            f.write(HoloSerializer.MAGIC)
            f.write(struct.pack('<I', HoloSerializer.VERSION))
            f.write(struct.pack('<I', num_layers))
            f.write(struct.pack('<I', num_heads))
            f.write(struct.pack('<I', head_dim))
            f.write(struct.pack('<I', rank))
            f.write(struct.pack('<I', seq_len))
            
            # Payload
            for i in range(num_layers):
                u = u_list[i].astype(np.float16)
                s = s_list[i].astype(np.float16)
                v = v_list[i].astype(np.float16)
                
                # Assume shapes are standard [NumHeads, ...]
                # Iterate each head to keep ordering simple
                for h in range(num_heads):
                    # Write U[h]
                    f.write(u[h].tobytes())
                    # Write S[h]
                    f.write(s[h].tobytes())
                    # Write V[h]
                    f.write(v[h].tobytes())
        
        print(f"Saved hologram to {filepath}")
