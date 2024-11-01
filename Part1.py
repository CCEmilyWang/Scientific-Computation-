 

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
            TimeTest1(method1,sorted_Ids_WithoutIndex,target_ids,None)) #(n,m)组合下用第一种方法去测试方法一的时间
        time_data["method1"]["test2"]["sorted"].append(
            TimeTest2(method1,sorted_Ids_WithoutIndex,target_ids,None)/test_runs)#(n,m)组合下用第2种方法去测试方法一的时间
        time_data["method2"]["test1"]["sorted"].append(
            TimeTest1(method2,sorted_Ids_WithIndex,target_ids,False)) #(n,m)组合下用第一种方法去测试方法2的时间
        time_data["method2"]["test2"]["sorted"].append(
            TimeTest2(method2,sorted_Ids_WithIndex,target_ids,False)/test_runs)#(n,m)组合下用第2种方法去测试方法2的时间
    
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

    '''
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
    '''
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





    '''






    M = [50,200,800,1000,5000,12000]
    N = [2000]*len(M) 
    # M = [2,4,6,8,10,12,14]
    

    counters = [[], [], []]  # C10, C20, C21
    # 使用字典存储方法、测试和排序组合的时间
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

    # 使用时，例如：
    # time_data["method1"]["test1"]["unsorted"].append(测试结果)

    

    for i, (n, m) in enumerate(zip(N, M)):
        # n = 50*n
        # m = 5*m
        print("n=",n,"m=",m)

        all_ids = random.sample(range(1,n*2), n)  
        target_ids = random.choices(range(1, n*2), k=m)

        time_data["method1"]["test1"]["unsorted"].append(
            TimeTest1(method1,all_ids,target_ids,None)) #(n,m)组合下用第一种方法去测试方法一的时间
        time_data["method1"]["test2"]["unsorted"].append(
            TimeTest2(method1,all_ids,target_ids,None)/test_runs)#(n,m)组合下用第一种方法去测试方法一的时间
        time_data["method2"]["test1"]["unsorted"].append(
            TimeTest1(method2,all_ids,target_ids,True)) #(n,m)组合下用第一种方法去测试方法2的时间
        time_data["method2"]["test2"]["unsorted"].append(
            TimeTest2(method2,all_ids,target_ids,True)/test_runs)#(n,m)组合下用第2种方法去测试方法2的时间


        import math



        # counters[0].append(n*m) #(n,m)组合下用方法一,时间复杂度=O(n*m)
        counters[0].append(m) #(n,m)组合下用方法一,时间复杂度=O(n*m)
        counters[1].append(n * math.log2(n)+m * math.log2(n)) #(n,m)组合下用方法一,时间复杂度=O(n*m)
        counters[2].append(m * math.log2(n)) #(n,m)组合下用方法一,时间复杂度=O(n*m)

        #处理排序后的列表
        Ids_WithIndex = list(enumerate(all_ids))
        sorted_Ids_WithIndex = sorted(Ids_WithIndex, key=lambda x: x[1])
        sorted_Ids_WithoutIndex = [_ for i,_ in sorted_Ids_WithIndex]
        # sorted_Ids_WithoutIndex = sorted(all_ids)

        time_data["method1"]["test1"]["sorted"].append(
            TimeTest1(method1,sorted_Ids_WithoutIndex,target_ids,None)) #(n,m)组合下用第一种方法去测试方法一的时间
        time_data["method1"]["test2"]["sorted"].append(
            TimeTest2(method1,sorted_Ids_WithoutIndex,target_ids,None)/test_runs)#(n,m)组合下用第2种方法去测试方法一的时间
        time_data["method2"]["test1"]["sorted"].append(
            TimeTest1(method2,sorted_Ids_WithIndex,target_ids,False)) #(n,m)组合下用第一种方法去测试方法2的时间
        time_data["method2"]["test2"]["sorted"].append(
            TimeTest2(method2,sorted_Ids_WithIndex,target_ids,False)/test_runs)#(n,m)组合下用第2种方法去测试方法2的时间



    plt.figure(2)

    data = [
        (counters[0], time_data["method1"]["test1"]["unsorted"], 'T_method1_test1_unsorted vs n*m', 'o'),
        (counters[0], time_data["method1"]["test2"]["unsorted"], 'T_method1_test2_unsorted vs n*m', 'x'),
        (counters[0], time_data["method1"]["test1"]["sorted"], 'T_method1_test1_sorted vs n*m', 'v'),
        (counters[0], time_data["method1"]["test2"]["sorted"], 'T_method1_test2_sorted vs n*m', '^'),
        (counters[0], time_data["method2"]["test1"]["unsorted"], 'T_method2_test1_unsorted vs n*m', 'o'),
        (counters[0], time_data["method2"]["test2"]["unsorted"], 'T_method2_test2_unsorted vs n*m', 'x'),
        (counters[0], time_data["method2"]["test1"]["sorted"], 'T_method2_test1_sorted vs n*m', 'v'),
        (counters[0], time_data["method2"]["test2"]["sorted"], 'T_method2_test2_sorted vs n*m', '^')
    ]
    data1 = [
        (counters[1], time_data["method2"]["test1"]["unsorted"], 'T_method2_test1_unsorted vs n*m', 'o'),
        (counters[1], time_data["method2"]["test2"]["unsorted"], 'T_method2_test2_unsorted vs n*m', 'x'),
        
    ]
    data2 = [
        (counters[2], time_data["method2"]["test1"]["sorted"], 'T_method2_test1_sorted vs n*m', 'v'),
        (counters[2], time_data["method2"]["test2"]["sorted"], 'T_method2_test2_sorted vs n*m', '^')

    ]

    # 使用循环绘制所有图形
    for x, y, label, marker in data:
        plt.plot(x, y, label=label, marker=marker)

    # 添加标题和标签
    plt.title('method: t and n * m')
    plt.xlabel('n * m')
    plt.ylabel('t')
    plt.yscale('log')

    # 添加图例
    plt.legend()

    # # 显示图形
    plt.grid()

    plt.show()
    '''

    '''

    plt.figure(1)
# 使用循环绘制所有图形
    for x, y, label, marker in data1:
        plt.plot(x, y, label=label, marker=marker)

    # 添加标题和标签
    plt.title('method: t and n * m')
    plt.xlabel('n * m')
    plt.ylabel('t')

    # 添加图例
    plt.legend()

    # # 显示图形
    plt.grid()

    plt.show()

    plt.figure(2)
    # 使用循环绘制所有图形
    for x, y, label, marker in data2:
        plt.plot(x, y, label=label, marker=marker)

    # 添加标题和标签
    plt.title('method: t and n * m')
    plt.xlabel('n * m')
    plt.ylabel('t')

    # 添加图例
    plt.legend()

    # # 显示图形
    plt.grid()

    plt.show()
    '''






















    '''


    #########3#33


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
    p = 2
    N = [2]  # 初始化列表，包含第一个元素
    for _ in range(1, 10):
        N.append(p*N[-1]**(p-1))   # 下一个元素



    # N = [4,6,8,10,12,14,16]
    # M = [2,4,6,8,10,12,14]
    M = [2 for _ in range(len(N))]

    for i, (n, m) in enumerate(zip(N, M)):
        p = 2
        # n = 20*n
        # # m = 20*m
        # # n1 = n**p
        # # m1 = m
        print("n=",n,"m=",m)
        all_ids = random.sample(range(1,n*2), n)  
        target_ids = random.sample(range(1, n*2), m)

        time_data["method1"]["test1"]["unsorted"].append(
            TimeTest1(method1,all_ids,target_ids,None)) #(n,m)组合下用第一种方法去测试方法一的时间
        time_data["method1"]["test2"]["unsorted"].append(
            TimeTest2(method1,all_ids,target_ids,None)/test_runs)#(n,m)组合下用第一种方法去测试方法一的时间
        time_data["method2"]["test1"]["unsorted"].append(
            TimeTest1(method2,all_ids,target_ids,True)) #(n,m)组合下用第一种方法去测试方法2的时间
        time_data["method2"]["test2"]["unsorted"].append(
            TimeTest2(method2,all_ids,target_ids,True)/test_runs)#(n,m)组合下用第2种方法去测试方法2的时间

       
               
        counters[1].append(n) #(n,m)组合下用方法2,时间复杂度=O(n*log2 n)


    p = 2
    M = [2]  # 初始化列表，包含第一个元素
    for _ in range(1, 10):
        M.append(p*M[-1]**(p-1))   # 下一个元素
    N = [1000 for _ in range(len(M))]

    for i, (n, m) in enumerate(zip(N, M)):
        print("n=",n,"m=",m)
        all_ids = random.sample(range(1,n*2), n)  
        target_ids = random.sample(range(1, n*2), m)

        #处理排序后的列表
        Ids_WithIndex = list(enumerate(all_ids))
        sorted_Ids_WithIndex = sorted(Ids_WithIndex, key=lambda x: x[1])
        sorted_Ids_WithoutIndex = [_ for i,_ in sorted_Ids_WithIndex]
        # sorted_Ids_WithoutIndex = sorted(all_ids)
        time_data["method1"]["test1"]["sorted"].append(
            TimeTest1(method1,sorted_Ids_WithoutIndex,target_ids,None)) #(n,m)组合下用第一种方法去测试方法一的时间
        time_data["method1"]["test2"]["sorted"].append(
            TimeTest2(method1,sorted_Ids_WithoutIndex,target_ids,None)/test_runs)#(n,m)组合下用第2种方法去测试方法一的时间
        time_data["method2"]["test1"]["sorted"].append(
            TimeTest1(method2,sorted_Ids_WithIndex,target_ids,False)) #(n,m)组合下用第一种方法去测试方法2的时间
        time_data["method2"]["test2"]["sorted"].append(
            TimeTest2(method2,sorted_Ids_WithIndex,target_ids,False)/test_runs)#(n,m)组合下用第2种方法去测试方法2的时间

   
        counters[2].append(m) #(n,m)组合下用方法2,时间复杂度=O(n*m)






    def plotting(i,data,title,xlabel):
        plt.figure(i)
        # 使用循环绘制所有图形
        for x, y, label, marker in data:
            plt.plot(x, y, label=label, marker=marker)

        # 添加标题和标签
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('t')

        # 添加图例
        plt.legend()
        # # 显示图形
        plt.grid()
        plt.show()

    # title=['method 1 : t and n * m',
    #        'method 2 : t and n',
    #        'method 2(sorted) : t and m',
    #        'method 1 and 2 : t and n * m']
    # title=['method 1 : t and n * m',
    #        'method 2 : t and n',
    #        'method 2(sorted) : t and m',
    #        'method 1 and 2 : t and n * m']
    # for i in range(4):
    #     plotting(i,data[i],title[i],xlabel[i])
        
        # 创建一个新图形
    




    plt.figure(4)

    # 创建一个列表，包含绘图所需的数据和标签
    data = [
        (counters[1], time_data["method1"]["test1"]["unsorted"], 'T_method1_test1_unsorted vs n*m', 'o'),
        (counters[1], time_data["method1"]["test2"]["unsorted"], 'T_method1_test2_unsorted vs n*m', 'x'),
        (counters[1], time_data["method1"]["test1"]["sorted"], 'T_method1_test1_sorted vs n*m', 'v'),
        (counters[1], time_data["method1"]["test2"]["sorted"], 'T_method1_test2_sorted vs n*m', '^'),
        (counters[1], time_data["method2"]["test1"]["unsorted"], 'T_method2_test1_unsorted vs n*m', 'o'),
        (counters[1], time_data["method2"]["test2"]["unsorted"], 'T_method2_test2_unsorted vs n*m', 'x'),
        (counters[1], time_data["method2"]["test1"]["sorted"], 'T_method2_test1_sorted vs n*m', 'v'),
        (counters[1], time_data["method2"]["test2"]["sorted"], 'T_method2_test2_sorted vs n*m', '^')

    ]

    # 使用循环绘制所有图形
    for x, y, label, marker in data:
        plt.plot(x, y, label=label, marker=marker)
        # # 设置x轴为对数坐标
        # plt.xscale('log')

        # # 自定义x轴刻度
        # plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    #     plt.xticks([1,2,3,4,5,6,7,8,9,10, 100, 1000], ['1', '2', '3', '4', '5', '6', '7', '8', '9','10', '100', '1000'])
    # # 添加标题和标签
    plt.title('method2 : t and n * m')
    plt.xlabel('n * m')
    plt.ylabel('t')

    # 添加图例
    plt.legend()

    # # 显示图形
    plt.grid()
    plt.show()

    

    plt.figure(5)

    # # 绘制 T00 相对于 n*m 的关系
    # plt.plot(C00,T00, label='T00 vs n*m',marker='o')

    # # 绘制 T01 相对于 n*m 的关系
    # plt.plot(C00,T01, label='T01 vs n*m', marker='x')

    # # 绘制 T00 相对于 n*m 的关系
    # plt.plot(C00,T10, label='T10 vs n*m',marker='v')

    # # 绘制 T01 相对于 n*m 的关系
    # plt.plot(C00,T11, label='T11 vs n*m', marker='^')
    # 创建一个列表，包含绘图所需的数据和标签
    data = [
        (counters[2], time_data["method1"]["test1"]["unsorted"], 'T_method1_test1_unsorted vs n*m', 'o'),
        (counters[2], time_data["method1"]["test2"]["unsorted"], 'T_method1_test2_unsorted vs n*m', 'x'),
        (counters[2], time_data["method1"]["test1"]["sorted"], 'T_method1_test1_sorted vs n*m', 'v'),
        (counters[2], time_data["method1"]["test2"]["sorted"], 'T_method1_test2_sorted vs n*m', '^'),
        (counters[2], time_data["method2"]["test1"]["unsorted"], 'T_method2_test1_unsorted vs n*m', 'o'),
        (counters[2], time_data["method2"]["test2"]["unsorted"], 'T_method2_test2_unsorted vs n*m', 'x'),
        (counters[2], time_data["method2"]["test1"]["sorted"], 'T_method2_test1_sorted vs n*m', 'v'),
        (counters[2], time_data["method2"]["test2"]["sorted"], 'T_method2_test2_sorted vs n*m', '^')

    ]

    # 使用循环绘制所有图形
    for x, y, label, marker in data:
        plt.plot(x, y, label=label, marker=marker)

    # 添加标题和标签
    plt.title('method2 : t and n * m')
    plt.xlabel('n * m')
    plt.ylabel('t')

    # 添加图例
    plt.legend()

    # # 显示图形
    plt.grid()
    plt.show()

    





    # for id in target_ids:
    #     time_start_m1 = time.time()
    #     method1(all_ids,id) 
    #     time_end_m1 = time.time()
    #     method2(all_ids,id,flag=True)
    #     print("method1",method1(all_ids,id))
    #     print("method2",method2(all_ids,id,flag=True))


    # indexed_ids = list(enumerate(all_ids))

    # sorted_indexed_ids = sorted(indexed_ids, key=lambda x: x[1])

    # for id in target_ids:
    #     method2(sorted_indexed_ids,id,flag=False)
    #     print("method2",method2(sorted_indexed_ids,id,flag=False))

    
    # print(all_ids)
    # print(target_ids)

    



    '''

    outputs=None
    return outputs

part1_test()

