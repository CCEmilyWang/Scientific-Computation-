 

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

    # 设置随机数种子
    random.seed(31)

    test_runs=100
    # Set value for n and m
    # n = random.randint(1,100)
    # m = random.randint(1,100)

    def use_method1 (all_ids,target_ids):
        for id in target_ids:
            method1(all_ids,id) 
    def use_method2(all_ids,target_ids,flag):
        for id in target_ids:
            method2(all_ids,id,flag) 

    #use time() to test time
    def TimeTest1(method,all_ids,target_ids,flag):
        if method == method1:
            t1 = time.perf_counter()
            use_method1(all_ids,target_ids)
            t2 = time.perf_counter()
        else :
            t1 = time.perf_counter()
            use_method2(all_ids,target_ids,flag)
            t2 = time.perf_counter()
        return t2-t1
    
    #use timeit() to test time
    def TimeTest2(method,all_ids,target_ids,flag):
        # 测量代码执行时间
        if method == method1:
            t = timeit.timeit(lambda:use_method1(all_ids,target_ids), number=test_runs)
        else:
            t = timeit.timeit(lambda:use_method2(all_ids,target_ids,flag), number=test_runs)
        return t

    N = [4,6,8,10,12,14,16]
    M = [2,4,6,8,10,12,14]
    #Tij:use TimeTesti to test time using methodj
    T11 = [] #use test1_time_method to test time using method 1
    T21 = [] #use test2_time_method to test time using method 1
    T11_sort = [] #use test1_time_method to test time using method 1
    T21_sort = [] #use test2_time_method to test time using method 1

    T12 = [] #use test1_time_method to test time using method 2
    T22 = [] #use test2_time_method to test time using method 2
    T12_sort = [] #use test1_time_method to test time using method 2
    T22_sort = [] #use test2_time_method to test time using method 2


    C10 = []
    C20 = []
    C21 = []

    
    #ti_j_k:= 用第i个测时间的方法去测试第j个方案，规模是第k种
    for i, (n, m) in enumerate(zip(N, M)):
        # n = 50*n
        # m = 20*m
        print("n=",n,"m=",m)
        all_ids = random.sample(range(1,n*2), n)  
        target_ids = random.sample(range(1, n*2), m)
        T11.append(TimeTest1(method1,all_ids,target_ids,None)) #(n,m)组合下用第一种方法去测试方法一的时间
        T21.append(TimeTest2(method1,all_ids,target_ids,None)/test_runs)#(n,m)组合下用第一种方法去测试方法一的时间
        
        C10.append(n*m) #(n,m)组合下用方法一,时间复杂度=O(n*m)

        #处理排序后的列表
        Ids_WithIndex = list(enumerate(all_ids))
        sorted_Ids_WithIndex = sorted(Ids_WithIndex, key=lambda x: x[1])
        sorted_Ids_WithoutIndex = [_ for i,_ in sorted_Ids_WithIndex]
        # sorted_Ids_WithoutIndex = sorted(all_ids)

        T11_sort.append(TimeTest1(method1,sorted_Ids_WithoutIndex,target_ids,None)) #(n,m)组合下用第一种方法去测试方法一的时间
        T21_sort.append(TimeTest2(method1,sorted_Ids_WithoutIndex,target_ids,None)/test_runs)#(n,m)组合下用第2种方法去测试方法一的时间
    plt.figure(1)

    data = [
        (C10, T11, 'T11 vs n*m', 'o'),
        (C10, T21, 'T21 vs n*m', 'x'),
        (C10, T11_sort, 'T11_sort vs n*m', 'v'),
        (C10, T11_sort, 'T21_sort vs n*m', '^')
    ]

    # 使用循环绘制所有图形
    for x, y, label, marker in data:
        plt.plot(x, y, label=label, marker=marker)

    # 添加标题和标签
    plt.title('method1 : t and n * m')
    plt.xlabel('n * m')
    plt.ylabel('t')

    # 添加图例
    plt.legend()

    # # 显示图形
    plt.grid()

    # plt.show()




    T11 = [] #use test1_time_method to test time using method 1
    T21 = [] #use test2_time_method to test time using method 1
    T11_sort = [] #use test1_time_method to test time using method 1
    T21_sort = [] #use test2_time_method to test time using method 1

    
    T12 = [] #use test1_time_method to test time using method 2
    T22 = [] #use test2_time_method to test time using method 2
    T12_sort = [] #use test1_time_method to test time using method 2
    T22_sort = [] #use test2_time_method to test time using method 2

    C20 = []
    C21 = []
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

        T11.append(TimeTest1(method1,all_ids,target_ids,None)) #(n,m)组合下用第一种方法去测试方法一的时间
        T21.append(TimeTest2(method1,all_ids,target_ids,None)/test_runs)#(n,m)组合下用第一种方法去测试方法一的时间
        

        T12.append(TimeTest1(method2,all_ids,target_ids,True)) #(n,m)组合下用第一种方法去测试方法2的时间
        T22.append(TimeTest2(method2,all_ids,target_ids,True)/test_runs)#(n,m)组合下用第2种方法去测试方法2的时间

               
        C20.append(n) #(n,m)组合下用方法2,时间复杂度=O(n*log2 n)


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
        T11_sort.append(TimeTest1(method1,sorted_Ids_WithoutIndex,target_ids,None)) #(n,m)组合下用第一种方法去测试方法一的时间
        T21_sort.append(TimeTest2(method1,sorted_Ids_WithoutIndex,target_ids,None)/test_runs)#(n,m)组合下用第2种方法去测试方法一的时间
   
        
        T12_sort.append(TimeTest1(method2,sorted_Ids_WithIndex,target_ids,False)) #(n,m)组合下用第一种方法去测试方法2的时间
        T22_sort.append(TimeTest2(method2,sorted_Ids_WithIndex,target_ids,False)/test_runs)#(n,m)组合下用第2种方法去测试方法2的时间

        C21.append(m) #(n,m)组合下用方法2,时间复杂度=O(n*m)






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
    




    plt.figure(2)

    # 创建一个列表，包含绘图所需的数据和标签
    data = [
        (C21, T11_sort, 'T11_s vs m', 'o'),
        (C21, T21_sort, 'T21_s vs ', 'x'),
        (C21, T12_sort, 'T12_sort vs m', 'v'),
        (C21, T22_sort, 'T22_sort vs m', '^')
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

    

    plt.figure(3)

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
        (C20, T21, 'T21 vs n', 'o'),
        (C20, T22, 'T22 vs n', 'x'),
        (C20, T11, 'T11 vs n', 'v'),
        (C20, T12, 'T12 vs n', '^')
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


    outputs=None
    return outputs

part1_test()