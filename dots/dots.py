import torch as t
import torch.nn.functional as f
import torch.nn.utils.stateless as stateless
from einops import rearrange
from dots.utils import entropy, get_device

def jacobian(model, inputs):
    # def input_as_fn_of_params(param_tensor):
    #     self.load_param_tensor(param_tensor)
    #     return self.forward(inputs)
    # return jacobian(input_as_fn_of_params, self.get_param_tensor())
    def input_as_fn_of_params(params_tensor):
        param_state_dict = model.param_tensor_to_state_dict(params_tensor)
        return stateless.functional_call(model, param_state_dict, inputs)
    return t.autograd.functional.jacobian(
        input_as_fn_of_params, model.get_param_tensor())

def matrix_jacobian(model, inputs):
    J = jacobian(model, inputs)
    if len(J.shape) == 3:
        J = rearrange(J, "b o p -> (b o) p")
    return J

def jacobian_matrix_rank(model, inputs):
    return t.linalg.matrix_rank(matrix_jacobian(model, inputs))

def jacobian_singular_values(model, inputs):
    return t.linalg.svd(matrix_jacobian(model, inputs)).S

def singular_value_rank(model, inputs, method="entropy"):
    if method=="entropy":
        return t.exp(entropy(jacobian_singular_values(model, inputs)))
    elif method=="heuristic":
        svs = jacobian_singular_values(model, inputs)
        # we create an array containing the sums of all values after that index:
        cumsums = t.flip(
            t.cumsum(
                t.flip(svs, dims=[0]),
                dim=0),
            dims=[0]
        ) - svs
        # find the first index i such that svs[i] > cumsums[i+1]:
        # (.float() because argmax not implemented for Bool tensors)
        # (NB: F I R S T  such index; there might be many)
        # (+1 because of 0-indexing)
        return t.argmax(((svs - cumsums) > 0).float()).item() + 1
    else:
        raise Exception(
            f"singular_value_rank does not implement method: {method}")

def jacobian_parameter_importances(model, inputs):
    jacobian = matrix_jacobian(model, inputs)
    return (jacobian ** 2).sum(dim=0)


class JModule(t.nn.Module):
    # To do Jacobians nicely, it is very convenient to have
    # a model class where you can get a tensor of all parameters.
    # This is what JModule does. All the functions above assume
    # that the model passed into them is a JModule
    # (rather than just any torch.nn.Module)
    def __init__(self):
        super().__init__()
        self.to(device=get_device())

    def get_param_tensor(self):
        return t.cat([
            parameter.view(-1)
            for parameter in self.parameters()])
    
    def param_tensor_to_state_dict(self, param_tensor):
        assert len(param_tensor.shape) == 1, "load_param_tensor expects 1D tensor"
        state_dict = self.state_dict()
        i = 0
        for key, parameter in state_dict.items():
            n_params = t.prod(t.tensor(parameter.shape))
            reshaped = param_tensor[i : i + n_params].view(parameter.shape)
            state_dict[key] = reshaped
            i += n_params
        return state_dict
    
    def load_param_tensor(self, param_tensor):
        state_dict = self.param_tensor_to_state_dict(param_tensor)
        self.load_state_dict(state_dict)
    
    def count_params(self):
        return self.get_param_tensor().shape[0]
    
    def jacobian(self, inputs):
        return jacobian(self, inputs)
    
    def matrix_jacobian(self, inputs):
        return matrix_jacobian(self, inputs)
    
    def jacobian_matrix_rank(self, inputs):
        return jacobian_matrix_rank(self, inputs)
    
    def jacobian_singular_values(self, inputs):
        return jacobian_singular_values(self, inputs)
    
    def singular_value_rank(self, inputs, method="entropy"):
        return singular_value_rank(self, inputs, method=method)
    
    def jacobian_parameter_importances(self, inputs):
        return jacobian_parameter_importances(self, inputs)
    
    

