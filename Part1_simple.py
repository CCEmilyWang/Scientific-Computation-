 

#----------------- Code for Part 1 -----------------#
def func2A(L,R):
    """
    Called by func2B
    """
    M = [] 
    indL,indR = 0,0
    nL,nR = len(L),len(R)

    for i in range(nL+nR):
        if L[indL][1]<R[indR][1]:
            M.append(L[indL])
            indL = indL + 1
            if indL>=nL:
                M.extend(R[indR:])
                break
        else:
            M.append(R[indR])
            indR = indR + 1
            if indR>=nR:
                M.extend(L[indL:])
                break
    return M

def func2B(X):
    """
    Called by method2
    """
    n = len(X)
    if n==1:
        return X
    else:
        L = func2B(X[:n//2])
        R = func2B(X[n//2:])
        return func2A(L,R)

def func2C(L,x):
    """
    Called by method2
    """
    istart = 0
    iend = len(L)-1

    while istart<=iend:
        imid = int(0.5*(istart+iend))
        if x==L[imid][1]:
            return L[imid][0]
        elif x < L[imid][1]:
            iend = imid-1
        else:
            istart = imid+1

    return -1000 

def method1(L,x):
    for ind,l in enumerate(L):
        if x==l:
            return ind        
    return -1000

def method2(L,x,flag=True):
    if flag:
        L2 = list(enumerate(L))
        Lnew = func2B(L2)
        return func2C(Lnew,x),Lnew
    else:
        return func2C(L,x)
        



import random
import timeit
import time
import matplotlib.pyplot as plt
import numpy as np

# Set the random seed
random.seed(31)
test_runs = 100

# Define functions to call methods
def use_method1(all_ids, target_ids):
    for id in target_ids:
        method1(all_ids, id)

def use_method2(all_ids, target_ids, flag):
    for id in target_ids:
        method2(all_ids, id, flag)

# Generalized time measurement function
def measure_time(method, all_ids, target_ids, flag, use_timeit=False):
    if method == method1:
        func = lambda: use_method1(all_ids, target_ids)
    else:
        func = lambda: use_method2(all_ids, target_ids, flag)
    
    if use_timeit:
        return timeit.timeit(func, number=test_runs) / test_runs
    else:
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        return end - start

# Function to plot the performance chart
def plot_performance(counters, time_data, complexities, title, xlabel):
    plt.figure()
    for label, data in time_data.items():
        plt.plot(counters, data, label=label, marker=data.get('marker', 'o'))
    for label, comp in complexities.items():
        plt.plot(counters, comp, label=label, linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('t(log)')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()

# Generate data and complexity values
def run_test(counters, N, M, is_fixed_n=True):
    time_data = {
        "T_method1_test1_unsorted": [], "T_method1_test2_unsorted": [],
        "T_method1_test1_sorted": [], "T_method1_test2_sorted": [],
        "T_method2_test1_unsorted": [], "T_method2_test2_unsorted": [],
        "T_method2_test1_sorted": [], "T_method2_test2_sorted": []
    }
    for n, m in zip(N, M):
        all_ids = random.sample(range(1, n*2), n)
        target_ids = random.choices(range(1, n*2), k=m)

        # Calculate the time for method 1 and method 2 in both unsorted and sorted cases
        time_data["T_method1_test1_unsorted"].append(measure_time(method1, all_ids, target_ids, None))
        time_data["T_method1_test2_unsorted"].append(measure_time(method1, all_ids, target_ids, None, use_timeit=True))
        time_data["T_method2_test1_unsorted"].append(measure_time(method2, all_ids, target_ids, True))
        time_data["T_method2_test2_unsorted"].append(measure_time(method2, all_ids, target_ids, True, use_timeit=True))

        # Sort for 'sorted' cases
        sorted_ids = sorted(all_ids)
        time_data["T_method1_test1_sorted"].append(measure_time(method1, sorted_ids, target_ids, None))
        time_data["T_method1_test2_sorted"].append(measure_time(method1, sorted_ids, target_ids, None, use_timeit=True))
        time_data["T_method2_test1_sorted"].append(measure_time(method2, sorted_ids, target_ids, False))
        time_data["T_method2_test2_sorted"].append(measure_time(method2, sorted_ids, target_ids, False, use_timeit=True))

        # Update counter
        counters.append(n if is_fixed_n else m)

    # Calculate theoretical complexity
    max_val = max(N if is_fixed_n else M)
    if is_fixed_n:
        complexities = {
            r'Theoretical Complexity $n$': N / (200 * max_val),
            r'Theoretical Complexity $n\log n + \log n$': N * np.log2(N) / (200 * max_val),
            r'Theoretical Complexity $\log n$': np.log2(N) / (200 * max_val)
        }
    else:
        complexities = {
            r'Theoretical Complexity $m$': M / (200 * max_val)
        }
    return time_data, complexities

# First chart (fixed m)
N_vals1 = np.array([1, 2, 3, 6, 9, 15, 17, 20, 50, 200, 800, 1000, 5000])
M_vals1 = [500] * len(N_vals1)
counters1 = []
time_data1, complexities1 = run_test(counters1, N_vals1, M_vals1)
plot_performance(counters1, time_data1, complexities1, 'Method Performance Comparison (Fix m)', 'n')

# Second chart (fixed n)
M_vals2 = np.array([1, 2, 3, 6, 9, 15, 17, 20, 50, 200, 800, 1000, 5000])
N_vals2 = [500] * len(M_vals2)
counters2 = []
time_data2, complexities2 = run_test(counters2, N_vals2, M_vals2, is_fixed_n=False)
plot_performance(counters2, time_data2, complexities2, 'Method Performance Comparison (Fix n)', 'm')

# Third chart (fixed r)
N_vals3 = np.array([1, 2, 3, 6, 9, 15, 17, 20, 50, 200, 800, 1000])
r = 10
M_vals3 = r * N_vals3
counters3 = []
time_data3, complexities3 = run_test(counters3, N_vals3, M_vals3)
plot_performance(counters3, time_data3, complexities3, 'Method Performance Comparison (Fix r = m/n)', 'n')
