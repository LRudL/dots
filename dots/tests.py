# %%
# import sys
# sys.path.append("..")
from models import MLP

# %%
model = MLP(1, 1, 2, hidden_size=6)

x = t.randn(4, 1)
print(f"Rank: {model.jacobian_matrix_rank(x)}")
print(f"SVDER: {model.singular_value_rank(x)}")
# %%
