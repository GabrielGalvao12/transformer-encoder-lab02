# lab_02 Transformer Encoder From Scratch
Implementação educacional do Encoder do Transformer usando apenas numpy.

# Baseado no artigo:
Attention Is All You Need (Vaswani et al., 2017)

# Requisitos
## Para o codigo funcionar é preciso executar no terminal a seguinte linha:
py -m pip install -r requirements.txt

## Executar o codigo rode no terminar a linha abaixo:
py test_encoder.py

# Estrutura do laboratorio:
attention.py =============> mecanismo de self attention
layer_norm.py ============> normalização
feed_forward.py ==========> rede densa
encoder.py ===============> camada encoder completa