# Adapt the FHE inference to use GL instead of CKKS

This means to adapt the current Fully Homomorphic Encryption (FHE) inference implementation to utilize the Gentry-Lin (GL) scheme instead of the Cheon-Kim-Kim-Song (CKKS) scheme. This involves modifying the encryption, decryption, and computation processes to align with the GL scheme's characteristics and requirements.

GL Scheme enables to encode matrices, where CKKS only authorizes encoding of vectors. This adaptation will require changes in how data is represented and processed during inference, ensuring that the operations are compatible with the GL scheme's capabilities.

**Files to be modified, in approximate appropriate order**:
- *bert.py* defines forward, dense, attention, KQV -> need to adapt to matrices (this should simplify the code)
- *model_encoder.py* -> encode weight matrices (same approach)
- *data.py* encrypt or encode data samples -> adapt data encoding to matrices (maybe do in batches)
- *linear.py* rotation and special vector operations -> adapt to matrices
- all *non-linear/* files describe non linear activation (norm, softmax, gelu...)


Check engine shape (slot_count, size, size) size=2^log_coeff_count:
(256, 16, 16)
(16, 256, 256)
(4, 1024, 1024)
(256, 32, 32)
(16, 512, 512)
(4, 2048, 2048)
(256, 64, 64)

**Problem** Desilo GL does not have bootstrapping