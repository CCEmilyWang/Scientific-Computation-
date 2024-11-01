""" 
    Your college id here:
    Template code for project 1, contains 7 functions:
    func2A, func2B, func2C, method1, method2: complete functions for part 1
    part1_test: function to be completed for part 1
    part2: function to be completed for question 2
"""


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
        


def part1_test(inputs=None):
    """Part 1, question 2: investigate trends in wall times of methods 1 and 2.
    Use the variables inputs and outputs if/as needed.
    You may import modules within this function as needed, please do not import
    modules elsewhere without permission.
    """
    import random
    import timeit
    import time
    import matplotlib.pyplot as plt
    import numpy as np


    random.seed(31)
    # Number of test runs
    test_runs=100
    # Function to apply method1 to each target_id
    def use_method1 (all_ids,target_ids):
        for id in target_ids:
            method1(all_ids,id) 
    # Function to apply method2 to each target_id with a flag
    def use_method2(all_ids,target_ids,flag):
        for id in target_ids:
            method2(all_ids,id,flag) 

    # Function to test execution time using time.perf_counter
    def TimeTest1(method,all_ids,target_ids,flag):
        if method == method1:
            t1 = time.perf_counter() # Start timer
            use_method1(all_ids,target_ids) # Execute method1
            t2 = time.perf_counter() # End timer
        else :
            t1 = time.perf_counter()
            use_method2(all_ids,target_ids,flag)
            t2 = time.perf_counter()
        return t2-t1 # Return elapsed time
    
    # Function to test execution time using timeit
    def TimeTest2(method,all_ids,target_ids,flag):
        # Measure code execution time
        if method == method1:
            t = timeit.timeit(lambda:use_method1(all_ids,target_ids), number=test_runs)
        else:
            t = timeit.timeit(lambda:use_method2(all_ids,target_ids,flag), number=test_runs)
        return t


    N = np.array([1, 2, 3, 6, 9, 15, 17, 20, 50, 200, 800, 1000, 5000])
    M = [500]*len(N)

    # Initialize counters for performance tracking
    counters = [[], [], []]  
    # Store execution times for different methods and tests in a dictionary
    time_data = {
        "method1": {
            "test1": {"unsorted": [], "sorted": []},
            "test2": {"unsorted": [], "sorted": []}
        },
        "method2": {
            "test1": {"unsorted": [], "sorted": []},
            "test2": {"unsorted": [], "sorted": []}
        }
    }

    for i, (n, m) in enumerate(zip(N, M)):
        # Generate random IDs
        all_ids = random.sample(range(1,n*2), n)  
        # Generate random target 
        target_ids = random.choices(range(1, n*2), k=m)
        # Test method1 and 2  with test1 and 2 for unsorted data
        time_data["method1"]["test1"]["unsorted"].append(
            TimeTest1(method1,all_ids,target_ids,None)) 
        time_data["method1"]["test2"]["unsorted"].append(
            TimeTest2(method1,all_ids,target_ids,None)/test_runs)
        time_data["method2"]["test1"]["unsorted"].append(
            TimeTest1(method2,all_ids,target_ids,True)) 
        time_data["method2"]["test2"]["unsorted"].append(
            TimeTest2(method2,all_ids,target_ids,True)/test_runs)
        
        counters[0].append(n) 
        # Calculate theoretical complexity values for method1 and method2
        complexity_values_method1 = N/(200*max(N))
        complexity_values_method2_unsorted = N * np.log2(N)/(200*max(N))
        complexity_values_method2_sorted =  np.log2(N)/(200*max(N))

        # Process sorted lists
        Ids_WithIndex = list(enumerate(all_ids))
        sorted_Ids_WithIndex = sorted(Ids_WithIndex, key=lambda x: x[1])
        sorted_Ids_WithoutIndex = [_ for i,_ in sorted_Ids_WithIndex]
        # Test method1 and 2  with test1 and 2 for sorted data
        time_data["method1"]["test1"]["sorted"].append(
            TimeTest1(method1,sorted_Ids_WithoutIndex,target_ids,None)) 
        time_data["method1"]["test2"]["sorted"].append(
            TimeTest2(method1,sorted_Ids_WithoutIndex,target_ids,None)/test_runs)
        time_data["method2"]["test1"]["sorted"].append(
            TimeTest1(method2,sorted_Ids_WithIndex,target_ids,False)) 
        time_data["method2"]["test2"]["sorted"].append(
            TimeTest2(method2,sorted_Ids_WithIndex,target_ids,False)/test_runs)
    
    plt.figure(1)
    data = [
    (counters[0], time_data["method1"]["test1"]["unsorted"], 'T_method1_test1_unsorted', 'o'),
    (counters[0], time_data["method1"]["test2"]["unsorted"], 'T_method1_test2_unsorted', '^'),
    (counters[0], time_data["method1"]["test1"]["sorted"], 'T_method1_test1_sorted', 's'),
    (counters[0], time_data["method1"]["test2"]["sorted"], 'T_method1_test2_sorted', 'D'),
    (counters[0], time_data["method2"]["test1"]["unsorted"], 'T_method2_test1_unsorted', 'v'),
    (counters[0], time_data["method2"]["test2"]["unsorted"], 'T_method2_test2_unsorted', '*'),
    (counters[0], time_data["method2"]["test1"]["sorted"], 'T_method2_test1_sorted', 'p'),
    (counters[0], time_data["method2"]["test2"]["sorted"], 'T_method2_test2_sorted', 'h'),
    (counters[0], complexity_values_method1, r'Theoretical Complexity $n$', 'o'),
    (counters[0], complexity_values_method2_unsorted, r'Theoretical Complexity $n\log n+ log n$', '^'),
    (counters[0], complexity_values_method2_sorted, r'Theoretical Complexity $log n$','s'),


]

    for x, y, label, marker in data:
        plt.plot(x, y, label=label, marker=marker)
    plt.title('Method Performance Comparison (Fix m)')
    plt.xlabel('n')
    plt.ylabel('t(log)')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.grid()
    plt.show()


    #####picture 2
    M = np.array([1,2,3,6,9,15,17,20,50, 200, 800, 1000,5000])
    
    N = [500]*len(M)

    time_data = {
        "method1": {
            "test1": {"unsorted": [], "sorted": []},
            "test2": {"unsorted": [], "sorted": []}
        },
        "method2": {
            "test1": {"unsorted": [], "sorted": []},
            "test2": {"unsorted": [], "sorted": []}
        }
    }

    for i, (n, m) in enumerate(zip(N, M)):
        all_ids = random.sample(range(1,n*2), n)  
        target_ids = random.choices(range(1, n*2), k=m)

        time_data["method1"]["test1"]["unsorted"].append(
            TimeTest1(method1,all_ids,target_ids,None)) 
        time_data["method1"]["test2"]["unsorted"].append(
            TimeTest2(method1,all_ids,target_ids,None)/test_runs)
        time_data["method2"]["test1"]["unsorted"].append(
            TimeTest1(method2,all_ids,target_ids,True)) 
        time_data["method2"]["test2"]["unsorted"].append(
            TimeTest2(method2,all_ids,target_ids,True)/test_runs)

        counters[1].append(m) 

        complexity_values = M/(200*max(M))
        
        Ids_WithIndex = list(enumerate(all_ids))
        sorted_Ids_WithIndex = sorted(Ids_WithIndex, key=lambda x: x[1])
        sorted_Ids_WithoutIndex = [_ for i,_ in sorted_Ids_WithIndex]

        time_data["method1"]["test1"]["sorted"].append(
            TimeTest1(method1,sorted_Ids_WithoutIndex,target_ids,None)) 
        time_data["method1"]["test2"]["sorted"].append(
            TimeTest2(method1,sorted_Ids_WithoutIndex,target_ids,None)/test_runs)
        time_data["method2"]["test1"]["sorted"].append(
            TimeTest1(method2,sorted_Ids_WithIndex,target_ids,False)) 
        time_data["method2"]["test2"]["sorted"].append(
            TimeTest2(method2,sorted_Ids_WithIndex,target_ids,False)/test_runs)

    plt.figure(2)
    data = [
    (counters[1], time_data["method1"]["test1"]["unsorted"], 'T_method1_test1_unsorted', 'o'),
    (counters[1], time_data["method1"]["test2"]["unsorted"], 'T_method1_test2_unsorted', '^'),
    (counters[1], time_data["method1"]["test1"]["sorted"], 'T_method1_test1_sorted', 's'),
    (counters[1], time_data["method1"]["test2"]["sorted"], 'T_method1_test2_sorted', 'D'),
    (counters[1], time_data["method2"]["test1"]["unsorted"], 'T_method2_test1_unsorted', 'v'),
    (counters[1], time_data["method2"]["test2"]["unsorted"], 'T_method2_test2_unsorted', '*'),
    (counters[1], time_data["method2"]["test1"]["sorted"], 'T_method2_test1_sorted', 'p'),
    (counters[1], time_data["method2"]["test2"]["sorted"], 'T_method2_test2_sorted', 'h'),
    (counters[1], complexity_values, r'Theoretical Complexity $m$', 'o')
    

]

    for x, y, label, marker in data:
        plt.plot(x, y, label=label, marker=marker)
    plt.title('Method Performance Comparison (Fix n)')
    plt.xlabel('m')
    plt.ylabel('t(log)')
    plt.yscale('log') 
    plt.legend()
    plt.grid()
    plt.show()

#######pictur 3
    N = np.array([1,2,3,6,9,15,17,20,50, 200, 800, 1000,5000])
    r=10
    M = r*N

    time_data = {
        "method1": {
            "test1": {"unsorted": [], "sorted": []},
            "test2": {"unsorted": [], "sorted": []}
        },
        "method2": {
            "test1": {"unsorted": [], "sorted": []},
            "test2": {"unsorted": [], "sorted": []}
        }
    }


    for i, (n, m) in enumerate(zip(N, M)):
        all_ids = random.sample(range(1,n*2), n)  
        target_ids = random.choices(range(1, n*2), k=m)

        time_data["method1"]["test1"]["unsorted"].append(
            TimeTest1(method1,all_ids,target_ids,None)) 
        time_data["method1"]["test2"]["unsorted"].append(
            TimeTest2(method1,all_ids,target_ids,None)/test_runs)
        time_data["method2"]["test1"]["unsorted"].append(
            TimeTest1(method2,all_ids,target_ids,True)) 
        time_data["method2"]["test2"]["unsorted"].append(
            TimeTest2(method2,all_ids,target_ids,True)/test_runs)


        counters[2].append(n) 
        
        complexity_values_method1 = N*M/(200*max(N))
        complexity_values_method2_unsorted = N * np.log2(N)+ M* np.log2(N)/(200*max(N))
        complexity_values_method2_sorted =  M * np.log2(N)/(200*max(N))
        
        Ids_WithIndex = list(enumerate(all_ids))
        sorted_Ids_WithIndex = sorted(Ids_WithIndex, key=lambda x: x[1])
        sorted_Ids_WithoutIndex = [_ for i,_ in sorted_Ids_WithIndex]

        time_data["method1"]["test1"]["sorted"].append(
            TimeTest1(method1,sorted_Ids_WithoutIndex,target_ids,None))
        time_data["method1"]["test2"]["sorted"].append(
            TimeTest2(method1,sorted_Ids_WithoutIndex,target_ids,None)/test_runs)
        time_data["method2"]["test1"]["sorted"].append(
            TimeTest1(method2,sorted_Ids_WithIndex,target_ids,False)) 
        time_data["method2"]["test2"]["sorted"].append(
            TimeTest2(method2,sorted_Ids_WithIndex,target_ids,False)/test_runs)
    

    plt.figure(3)
    data = [
    (counters[2], time_data["method1"]["test1"]["unsorted"], 'T_method1_test1_unsorted', 'o'),
    (counters[2], time_data["method1"]["test2"]["unsorted"], 'T_method1_test2_unsorted', '^'),
    (counters[2], time_data["method1"]["test1"]["sorted"], 'T_method1_test1_sorted', 's'),
    (counters[2], time_data["method1"]["test2"]["sorted"], 'T_method1_test2_sorted', 'D'),
    (counters[2], time_data["method2"]["test1"]["unsorted"], 'T_method2_test1_unsorted', 'v'),
    (counters[2], time_data["method2"]["test2"]["unsorted"], 'T_method2_test2_unsorted', '*'),
    (counters[2], time_data["method2"]["test1"]["sorted"], 'T_method2_test1_sorted', 'p'),
    (counters[2], time_data["method2"]["test2"]["sorted"], 'T_method2_test2_sorted', 'h'),
    (counters[2], complexity_values_method1, r'Theoretical Complexity $n*m$', 'o'),
    (counters[2], complexity_values_method2_unsorted, r'Theoretical Complexity $n\log n+ m\log n$', '^'),
    (counters[2], complexity_values_method2_sorted, r'Theoretical Complexity $m\log n$','s'),
    

]


    # 使用循环绘制所有图形
    for x, y, label, marker in data:
        plt.plot(x, y, label=label, marker=marker)

    # 添加标题和标签
    plt.title('Method Performance Comparison (Fix r=m/n)')
    plt.xlabel('n')
    plt.ylabel('t(log)')
    plt.yscale('log') 

    plt.legend()
    plt.grid()
    plt.show()



    outputs=None
    return outputs

part1_test()
#----------------- End code for Part 1 -----------------#


#----------------- Code for Part 2 -----------------#

def part2(A1,A2,L):
    """Part 2: Complete function to find amino acid patterns in
    amino acid sequences, A1 and A2
    Input:
        A1,A2: Length-n strings corresponding to amino acid sequences
        L: List of l sub-lists. Each sub-list contains 2 length-m strings. Each string corresponds to an amino acid sequence
        sequence
    Output:
        F: List of lists containing locations of amino-acid sequence pairs in A1 and A2.
        F[i] should be a list of integers containing all locations in A1 and A2 at
        which the amino acid sequence pair stored in L[i] occur in the same place.
    """

    
    # Map a to i to 1 to 9
    # Build hash function to convert A1, A2, and L into corresponding numbers
    # Loop through L


    n = len(A1) #A2 should be same length
    l = len(L)
    m = len(L[0][0])
    F = [[] for i in range(l)]
    base = 9
    # Large prime number for modulus
    mod = 10**9 + 7 
    # Reusable base^m % mod
    base_m = pow(base, m, mod)  
    # ASCII to integer mapping for a-i characters
    c2b = {chr(97 + j): j for j in range(9)}  

    def hash_function(A, m):
        """Compute hash values for all substrings of length m in A"""
        hashes = {}
        current_hash = 0
        # Compute the hash for the first m_substring
        for i in range(m):
            current_hash = (current_hash * base + c2b[A[i]]) % mod
        
        # Store first hash with start index
        hashes[current_hash] = [0]  

        # Use sliding window to calculate subsequent hashes
        for i in range(1, n - m + 1):
            current_hash = (current_hash * base - c2b[A[i-1]] * base_m + c2b[A[i + m - 1]]) % mod
            current_hash = (current_hash + mod) % mod  # Ensure hash value is non-negative
            if current_hash in hashes:
                hashes[current_hash].append(i)  # If hash value exists, add starting index
            else:
                hashes[current_hash] = [i]  # Otherwise, create a new entry


        return hashes


    # Hash A1 and A2
    hashes_A1 = hash_function(A1, m)
    hashes_A2 = hash_function(A2, m)


    def get_hash(sub):
        """Compute hash for a given substring"""
        current_hash = 0
        for i in range(m):
            current_hash = (current_hash * base + c2b[sub[i]]) % mod
        return current_hash
    
    # Iterate through each sublist in L
    for index, (sub1, sub2) in enumerate(L):
        hash_sub1 = get_hash(sub1) 
        hash_sub2 = get_hash(sub2)  

        positions = []
        # Only proceed if hashes match in both sequences
        if hash_sub1 in hashes_A1 and hash_sub2 in hashes_A2:
            #check the positions where the hashes match in A1
            for pos1 in hashes_A1[hash_sub1]:
                if A2[pos1:pos1 + m] == sub2:
                    positions.append(pos1)               
        F[index] = positions
    return F

#----------------- End code for Part 2 -----------------#



if __name__=='__main__':
    x=0 #please do not remove
    #Add code here to call part1_test and generate the figures included in your report.
