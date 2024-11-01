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
        for i in range(1, len(A) - m + 1):
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


