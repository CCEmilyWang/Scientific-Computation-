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
    # def get_ids_lists(num_lists, list_size, value_range):
    #     ids_lists = []
    #     for _ in range(num_lists):
    #         ids_list = [random.sample(range(value_range[0], value_range[1]) , list_size)for _ in range(list_size)]
    #         ids_lists.append(ids_list)
    #     return ids_lists

    # # 示例：生成5个列表，每个列表包含10个随机整数，范围在1到100之间
    # lists = get_ids_lists(5, 10, (1, 100))
    # print(lists)


    # 设置随机数种子
    random.seed(31)

    test_runs=100
    # Set value for n and m
    # n = random.randint(1,100)
    # m = random.randint(1,100)



    # a = [2,4,6,8,10]
    # Python list containing n integer IDs 
    # all_ids = random.sample(range(1,200), n)  
    # # a sequence of m integer IDs.
    # target_ids = random.sample(range(1, 200), m)
    # def run_method1():
    #     for id in target_ids:
    #         method1(all_ids, id)
    # execution_time = timeit.timeit(run_method1, number=1000)
    # print(f"Execution time: {execution_time} seconds")


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


    T = [] #use test2_time_method to test time
    T1 = [] #use test2_time_method to test time

    C10 = []
    C20 = []
    C21 = []

    
    #ti_j_k:= 用第i个测时间的方法去测试第j个方案，规模是第k种
    for i, (n, m) in enumerate(zip(N, M)):
        n = 50*n
        m = 20*m
        n1 = 10*n
        m1 = 5*m
        print("n=",n,"m=",m,"n1=",n1, "m1=", m1)
        all_ids = random.sample(range(1,n1*2), n1)  
        target_ids = random.sample(range(1, n1*2), m1)
        # 处理排序前的列表

        #t=test1_time_method(method1,all_ids[:n],target_ids[:m])  
        #t1=test1_time_method(method1,all_ids,target_ids)
        T11.append(TimeTest1(method1,all_ids[:n],target_ids[:m],None)) #(n,m)组合下用第一种方法去测试方法一的时间
        #t=test2_time_method(method1,all_ids[:n],target_ids[:m])
        #t1=test2_time_method(method1,all_ids,target_ids)
        T21.append(TimeTest2(method1,all_ids[:n],target_ids[:m],None)/test_runs)#(n,m)组合下用第一种方法去测试方法一的时间
        
        C10.append(n*m) #(n,m)组合下用方法一,时间复杂度=O(n*m)

        T12.append(TimeTest1(method2,all_ids[:n],target_ids[:m],True)) #(n,m)组合下用第一种方法去测试方法2的时间
        T22.append(TimeTest2(method2,all_ids[:n],target_ids[:m],True)/test_runs)#(n,m)组合下用第2种方法去测试方法2的时间

        #处理排序后的列表
        Ids_WithIndex = list(enumerate(all_ids))
        sorted_Ids_WithIndex = sorted(Ids_WithIndex, key=lambda x: x[1])
        sorted_Ids_WithoutIndex = [_ for i,_ in sorted_Ids_WithIndex]
        # sorted_Ids_WithoutIndex = sorted(all_ids)

        T11_sort.append(TimeTest1(method1,sorted_Ids_WithoutIndex[:n],target_ids[:m],None)) #(n,m)组合下用第一种方法去测试方法一的时间
        T21_sort.append(TimeTest2(method1,sorted_Ids_WithoutIndex[:n],target_ids[:m],None)/test_runs)#(n,m)组合下用第2种方法去测试方法一的时间


        import math 
        

        T12_sort.append(TimeTest1(method2,sorted_Ids_WithIndex[:n],target_ids[:m],False)) #(n,m)组合下用第一种方法去测试方法2的时间
        T22_sort.append(TimeTest2(method2,sorted_Ids_WithIndex[:n],target_ids[:m],False)/test_runs)#(n,m)组合下用第2种方法去测试方法2的时间


        C20.append((n+m)*math.log(n, 2)) #(n,m)组合下用方法一,时间复杂度=O(n*m)
        C21.append(math.log(n, 2)*m) #(n,m)组合下用方法一,时间复杂度=O(n*m)

        # print(n,m,t)
        # print(n1,m1,t1)
        # print("use test1_time_method to test time\n time ratio=",t1/t,"scale ratio=",n1*m1/(n*m),"coefficien=" ,n1*m1*t/(n*m*t1))
        # print("use test2_time_method to test time\n time ratio=",t11/t10,"scale ratio=",n1*m1/(n*m),"coefficien=" ,n1*m1*t10/(n*m*t11))

    def plotting(i,data,title):
        plt.figure()

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

        plt.show()


        
        # 创建一个新图形
    plt.figure(1)

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

    plt.show()




    plt.figure(2)

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
        # (C20, T12, 'T12 vs n*m', 'o'),
        # (C20, T22, 'T22 vs n*m', 'x'),
        (C21, T12_sort, 'T12_sort vs n*m', 'v'),
        (C21, T12_sort, 'T22_sort vs n*m', '^')
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

    #use/modify the code below as needed
    n = len(A1) #A2 should be same length
    l = len(L)
    m = len(L[0][0])
    F = [[] for i in range(l)]

    def char2base4(S):
        """Convert gene test_sequence
        string to list of ints
        """
        c2b = {chr(97 + j): j for j in range(10)}  # 97 是字符 'a' 的 ASCII 值
        L = [c2b.get(s) for s in S]
        return L
    
    X = char2base4(A1)
    Y = char2base4(A2)
    
    def heval(L,Base,Prime):
        """Convert list L to base-10 number mod Prime
        where Base specifies the base of L
        """
        f = 0
        for l in L[:-1]:
            f = Base*(l+f)
            h = (f + (L[-1])) % Prime
        return h
    def match():
        
    ind=0
    Base = m 
    Prime = n  ######
    hp = heval(Y,Base,Prime)
    imatch=[]
    hi = heval(X[:m],Base,Prime)
    if hi==hp:
        if match(X[:m],Y): #Character-by-character comparison
            imatch.append(ind)
    bm = (4**m) % q
    for ind in range(1,n-m+1):
        #Update rolling hash
        hi = (4*hi – int(X[ind-1])*bm + int(X[ind-1+m])) % q
        if hi==hp: #If hashes match, check if strings match
            if match(X[ind:ind+m],Y): imatch.append(ind)

  

#----------------- End code for Part 2 -----------------#



if __name__=='__main__':
    x=0 #please do not remove
    #Add code here to call part1_test and generate the figures included in your report.
