import csv
import datetime
import os
import re
import typing as t
from sympy import sympify, symbols, lambdify
from typing import List
from sklearn.metrics import r2_score

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
import numpy as np


from utils.data import get_dynamic_data
from utils.log_ import create_dir_if_not_exist, save_pareto_frontier_to_csv
import gc
import torch
from model.regressor import PSRN_Regressor

import warnings
warnings.filterwarnings("ignore")


file_idx =1
NOISE_LEVEL = 0.1 # 0.01, 0.1
if NOISE_LEVEL == 0.01:
    noise_name = "001"
elif NOISE_LEVEL == 0.1:
    noise_name = "01"
else:
    raise ValueError("NOISE_LEVEL must be 0.01 or 0.1")

os.environ['CUDA_VISIBLE_DEVICES'] = str("1")

def process_benchmark(expression):
    pow_regexp = r"pow\((.*?),(.*?)\)"
    pow_replace = r"((\1) ^ (\2))"
    processed = re.sub(pow_regexp, pow_replace, expression)

    div_regexp = r"div\((.*?),(.*?)\)"
    div_replace = r"((\1) / (\2))"
    processed = re.sub(div_regexp, div_replace, processed)
    # processed = processed.replace("x1", "x")
    # processed = processed.replace("x2", "y")
    return processed


def open_csv(file_name: str):
    equations = []
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            equations.append(
                (
                    row["name"],
                    int(row["variables"]),
                    process_benchmark(row["expression"]),
                    row["distribution"],
                    float(row["lb"]),
                    float(row["ub"]),
                    int(row["sample_num"]),
                )
            )
    return equations


def expr_to_func(sympy_expr, variables: List[str]):
    def cot(x):
        return 1 / np.tan(x)

    def acot(x):
        return 1 / np.arctan(x)

    def coth(x):
        return 1 / np.tanh(x)

    return lambdify(
        variables,
        sympy_expr,
        modules=["numpy", {"cot": cot, "acot": acot, "coth": coth}],
    )


def create_dataset(f, 
                   n_var=2, 
                   f_mode = 'col',
                   ranges = [-1,1],
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0,
                   func_type: str = 'np',
                   distribution="U"):
    '''
    cate dataset
    
    Args:
    -----
        f : function
            the symbolic formula used to cate the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    turns:
    --------
        dataset : dic
            Train/test inputs/labels a dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']
         
    Example
    -------
    >>> f = lambda x: np.exp(np.sin(np.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = cate_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    np.Size([100, 2])
    '''

    np.random.seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var,2)
    else:
        ranges = np.array(ranges)
        
    if func_type == 'np':
        train_input = np.zeros((train_num, n_var))
        test_input = np.zeros((test_num, n_var))
        for i in range(n_var):
            train_input[:,i] = np.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
            test_input[:,i] = np.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
    elif func_type == 'numpy':
        train_input = np.zeros((train_num, n_var))
        test_input = np.zeros((test_num, n_var))
        if distribution == "U":
            for i in range(n_var):
                train_input[:,i] = np.random.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
                test_input[:,i] = np.random.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        else:
            # 从ranges均匀采样train_num(test_num)个点
            for i in range(n_var):
                train_input[:,i] = np.linspace(ranges[i,0], ranges[i,1], train_num)
                test_input[:,i] = np.linspace(ranges[i,0], ranges[i,1], test_num)

                
    train_label = np.zeros((train_num, 1))
    test_label = np.zeros((test_num, 1))
    for i in range(train_num):
        # 我希望f能够根据n_var的数量来决定输入的数量,而不是直接输入一个向量test_input[i]
        train_label[i] = f(*list(train_input[i]))
    for i in range(test_num):
        test_label[i] = f(*list(test_input[i]))

    y_rms = ((train_label ** 2).mean()) ** 0.5
    epsilon = NOISE_LEVEL * np.random.normal(0, y_rms, len(train_label)).reshape(-1, 1)
    train_label = train_label + epsilon
        
    # if has only 1 dimension
    if len(train_label.shape) == 1:
        train_label = train_label.unsqueeze(dim=1)
        test_label = test_label.unsqueeze(dim=1)
        
    def normalize(data, mean, std):
            return (data-mean)/std
            
    if normalize_input == True:
        mean_input = np.mean(train_input, dim=0, keepdim=True)
        std_input = np.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
        
    if normalize_label == True:
        mean_label = np.mean(train_label, dim=0, keepdim=True)
        std_label = np.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset['train_input'] = train_input
    dataset['test_input'] = test_input

    dataset['train_label'] = train_label
    dataset['test_label'] = test_label

    return dataset


# @command("evaluate-benchmark")
def evaluate(
    benchmark_path: t.Optional[str] = None,
    repeat_times: int = 10,
):
    
    for i, (name, num_variables, eq, distribution, lb, ub, sample_num) in enumerate(open_csv(benchmark_path)):
        ops = ['Add','Mul','SemiSub','SemiDiv','Identity','Sign','Sin','Cos','Exp','Log']
        sym_eq = sympify(eq)
        num_variables = int(num_variables)
        variables = symbols("x,y")if num_variables == 2 else symbols("x")

        variables_str = ""
        variables_name = []
        for i in range(num_variables):
            variables_str += f"x{i+1}"
            variables_name.append(f"x{i+1}")
            if i != num_variables - 1:
                variables_str += ","
        variables = symbols(variables_str)

        n_inputs = 5

        if num_variables >= 3:
            # path = "dr_mask/3_9_[Add_Mul_Identity_Neg_Inv_Sin_Cos_Log]_mask.npy"
            n_inputs = 9
            ops = ['Add','Mul','Identity','Neg','Inv','Sin','Cos','Log']


        f = expr_to_func(sym_eq, variables=variables)
        dataset = create_dataset(f, n_var=num_variables, device="cpu", ranges=[lb, ub], 
                                 train_num=sample_num, func_type="numpy", distribution=distribution)
        for idx in range(repeat_times):
            prev = datetime.datetime.now()
            X = dataset["train_input"]
            y = dataset["train_label"]
            # to tensor
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            qsrn_regressor = PSRN_Regressor(variables=variables_name,
                                        operators=ops,
                                        n_symbol_layers=3,
                                        n_inputs=n_inputs,
                                        use_const=True,
                                        trying_const_num=3,
                                        trying_const_range=[-10,10],
                                        trying_const_n_try=1,
                                        device='cuda',
                                        )
            
            flag, pareto = qsrn_regressor.fit(X,
                                        y,
                                        n_down_sample=50,
                                        n_step_simulation=5,
                                        use_threshold=False,
                                        real_time_display_freq=1,
                                        prun_ndigit=3,
                                        top_k=10,
                                        add_bias=True,
                                        )
            
            "dr_mask/3_6_[Add_Mul_Identity_Pow2_Pow3_Inv_Neg_Cos_Cosh_Exp_Log_Sin_Tanh_Sqrt]_mask.npy"
            
            import sympy as sp

            crit = "mse"
            pareto_ls = qsrn_regressor.display_expr_table(sort_by=crit)
            expr_str, reward, loss, complexity = pareto_ls[0]
            expr_str_best_MSE = expr_str
            expr_sympy_best_MSE = sp.simplify(expr_str)

            eq_pred = expr_to_func(expr_sympy_best_MSE, variables=variables)

            # to numpy
            X = np.array(X)
            y = np.array(y)

            y_pred = np.zeros_like(y)
            for i in range(X.shape[0]):
                y_pred[i] = eq_pred(*X[i])

            r2 = r2_score(y, y_pred)
            mae = np.mean(np.abs(y - y_pred))

            print(idx)
            print(name)
            print(eq)
            print(r2, mae)
            print(datetime.datetime.now() - prev)
            info_dict = {
                "idx": idx,
                "name": name,
                "eq": str(expr_str),
                "r2": r2,
                "mae": mae,
            }
            # 将info_dict写入benchresult.csv文件,csv文件的表头为info_dict的key
            with open(f"benchresult_easy_{NOISE_LEVEL}_{file_idx}.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=info_dict.keys())
                writer.writerow(info_dict)


    # print(datetime.datetime.now() - starting_time)
    # print("\nResults:")
    # print(np.mean(r2s), np.median(r2s))
    # print(np.mean(maes), np.median(maes))


if __name__ == "__main__":
    evaluate(f"benchmarks_{file_idx}_{noise_name}.csv", repeat_times=10)
