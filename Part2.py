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

    # 将A1、A2、L变成对应数字
    # 构建哈希函数
    # 循环A1、A2 

    #use/modify the code below as needed
    n = len(A1) #A2 should be same length
    l = len(L)
    m = len(L[0][0])
    F = [[] for i in range(l)]
    base = 9
    mod = 11505703
    # mod = 10**9 + 7  # 大素数，用于模运算
    base_m = pow(base, m, mod)  # 计算 base^m % mod
    c2b = {chr(97 + j): j for j in range(9)}  # 97 是字符 'a' 的 ASCII 值

    print("nnnn",n)

    def hash_function(A, m):
        hashes = {}
        current_hash = 0
        # 计算第一个长度为 m 的子串的哈希值
        for i in range(m):
            current_hash = (current_hash * base + c2b[A[i]]) % mod

        hashes[current_hash] = [0]  # 存储第一个子串的哈希值和起始索引
        # hashes = {‘哈希值’：[初始位置]}
        #hashes = {‘123’：[0]}

        # 滑动窗口计算每个长度为 m 的子串的哈希值
        for i in range(1, len(A) - m + 1):
            current_hash = (current_hash * base - c2b[A[i-1]] * base_m + c2b[A[i + m - 1]]) % mod
            current_hash = (current_hash + mod) % mod  # 确保哈希值为非负
            if current_hash in hashes:
                hashes[current_hash].append(i)  # 如果哈希值已经存在，添加起始索引
            else:
                hashes[current_hash] = [i]  # 否则，创建新的条目

        return hashes


    # 计算 A1 和 A2 的哈希值
    hashes_A1 = hash_function(A1, m)
    hashes_A2 = hash_function(A2, m)


 # 计算子串的哈希值
    def get_hash(sub):
        current_hash = 0
        for i in range(m):
            current_hash = (current_hash * base + c2b[sub[i]]) % mod
        return current_hash

    for index, (sub1, sub2) in enumerate(L):
        hash_sub1 = get_hash(sub1)  # 使用自定义哈希函数计算哈希值
        hash_sub2 = get_hash(sub2)  # 使用自定义哈希函数计算哈希值

        positions = []
        if hash_sub1 in hashes_A1 and hash_sub2 in hashes_A2:
            for pos1 in hashes_A1[hash_sub1]:
                if A2[pos1:pos1 + m] == sub2:
                    positions.append(pos1)
                    # print(f"Matched position {pos1} for sub1: {sub1} and sub2: {sub2}")
                # else:
                #     print(f"No match at position {pos1} for sub2: {sub2}")
                    

        F[index] = positions
    print("-----------------\n", F)





    # def char2base4(S):
    #     """Convert gene test_sequence
    #     string to list of ints
    #     """
    
    #     c2b = {chr(97 + j): j for j in range(10)}  # 97 是字符 'a' 的 ASCII 值
    #     L = [c2b.get(s) for s in S]
    #     return L
    
    # X = char2base4(A1)
    # Y = char2base4(A2)
    
    # def heval(L,Base,Prime):
    #     """Convert list L to base-10 number mod Prime
    #     where Base specifies the base of L
    #     """
    #     f = 0
    #     for l in L[:-1]:
    #         f = Base*(l+f)
    #         h = (f + (L[-1])) % Prime
    #     return h
    # def match():
        
    # ind=0
    # Base = m 
    # Prime = n  ######
    # hp = heval(Y,Base,Prime)
    # imatch=[]
    # hi = heval(X[:m],Base,Prime)
    # if hi==hp:
    #     if match(X[:m],Y): #Character-by-character comparison
    #         imatch.append(ind)
    # bm = (4**m) % q
    # for ind in range(1,n-m+1):
    #     #Update rolling hash
    #     hi = (4*hi – int(X[ind-1])*bm + int(X[ind-1+m])) % q
    #     if hi==hp: #If hashes match, check if strings match
    #         if match(X[ind:ind+m],Y): imatch.append(ind)

  

#----------------- End code for Part 2 -----------------#






import random

def read_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content

def generate_random_sequence(length):
    # 生成一个随机字符串，长度为 length，字符从 'a' 到 'i'
    return ''.join(random.choice('abcdefghi') for _ in range(length))

def generate_L(num_pairs, m):
    L = []
    for _ in range(num_pairs):
        # 为每一对生成两个随机字符串
        sublist = [generate_random_sequence(m), generate_random_sequence(m)]
        L.append(sublist)
    return L

# # 生成 1000 对，长度为 3 的字符串
num_pairs = 1000
m = 3
L = generate_L(num_pairs, m)
print(L)
# L = [['adb', 'gcg'], ['dbi', 'cgh'], ['fdi', 'eah'], ['fcc', 'ieh'], ['bfh', 'ich']]

A1 = read_file('A1.txt')
A2 = read_file('A2.txt')

# A1 = "adbicacahbadbifefgedeaifecbifdieifdffdidcahgcgchifffcgcgchhhieideiffggefedcieagdicaaabeacaaefeebfebcahicbhieidcbcagcgfdgdiffffdhgaicbhhiddcahgbhga"
# A2 = "gcghgahhifcabffdfeahaeebgefeeahaahhagahicifciebcgcbchgcghgbchifgbcahidgeeahhhbgdcahgcahbicieieacagcagebidddcbggedebffdfeaabchbaahhifdcagdeicieahie"
part2(A1,A2,L)


