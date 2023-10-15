#Problem1

#Introduction

#Say "Hello, World!" With Python
if __name__ == '__main__':
    string="Hello, World!"
    print(string)


#Python If-Else
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    n = int(input().strip())
if(n>=1 and n<=100):
    if (n%2!=0):
        print("Weird")
    if(n%2==0 and n>=2 and n<=5):
        print("Not Weird")
    if(n%2==0 and n>=6 and n<=20):
        print("Weird")
    if(n>20 and n%2==0):
        print("Not Weird")


#Arithmetic Operations
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    if(a>=1 and a<=10**10 and b>=1 and b<=10**10):
        sum_=a+b
        diff=a-b
        prod=a*b
    print(sum_)
    print(diff)
    print(prod)   


#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    int_div=a//b
    float_div=a/b
    print(int_div)
    print(float_div)



#Loops
if __name__ == '__main__':
    n = int(input())
    i=0
    if(n>=1 and n<=20):
       while(i<n):
           print(i**2)
           i+=1


#Write a function
def is_leap(year):
    leap = False
    if(year>=1900 and year<=10**5):
        if(year%4==0):
            if(year%100==0):
                if(year%400==0):
                    leap = True
                else:
                    leap=False
            else:
                leap=True
    return leap
if __name__=='__main__':   
    year = int(input())
    print(is_leap(year))


#Print Function
if __name__ == '__main__':
    n = int(input())
    if(n>=1 and n<=150):
        for i in range(1,n+1):
            print(i, end='')




#Data Types

#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    mylist=[[i,j,k] for i in range(0,x+1) for j in range(0,y+1) for k in range(0,z+1)]
    finlist=[[i,j,k] for [i,j,k] in mylist if i+j+k!=n]
    print(finlist)


#Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    mylist=list(arr)
    uniqueval=set(mylist)
    sorted_list=sorted(uniqueval, reverse=True)
    runner_up=sorted_list[1]
    print(runner_up)


#Nested List
if __name__ == '__main__':
    nested_list=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        nested_list.append([name,score])   
    mylist=[]
    for i in nested_list:
        mylist.append(i[1])   
    mymylist=set(mylist)
    sorted_list=sorted(mymylist,reverse=False)
    elem=sorted_list[1]
    new_list=[]
    for i in nested_list:
        if (elem==i[1]):
            new_list.append(i[0])
    def_list=sorted(new_list)
    for i in def_list:
        print(i)


#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name=input()
    mean=sum(student_marks[query_name])/len(student_marks[query_name])
    defmean= f"{mean:.2f}"
    print(defmean)


#Lists 
if __name__ == '__main__':
   # Initialize an empty list
    my_list = []
# Read the number of commands
    N = int(input())
# Initialize a list to store the results
    results = []
# Read and perform the specified commands
    for _ in range(N):
        command = input().split()
        action = command[0]
        if action == "insert":
            i, e = map(int, command[1:])
            my_list.insert(i, e)
        elif action == "print":
            results.append(list(my_list))
        elif action == "remove":
            e = int(command[1])
            my_list.remove(e)
        elif action == "append":
            e = int(command[1])
            my_list.append(e)
        elif action == "sort":
            my_list.sort()
        elif action == "pop":
            my_list.pop()
        elif action == "reverse":
            my_list.reverse()
# Print the results
for result in results:
    print(result)


#Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=tuple(integer_list)
    print(hash(t))




#Strings

#sWAP cASE
def swap_case(s):
    modstring=""
    for i in s:
        if i.islower():
            modstring+=i.upper()
        elif i.isupper():
            modstring+=i.lower()
        else:
            modstring+=i
    return modstring
if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


#String Split and Join
def split_and_join(line):
    line=line.split(" ")
    line="-".join(line)
    return line
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


#What's Your Name?
def print_full_name(first, last):
    print(f"Hello {first} {last}! You just delved into python. ")
if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


#Mutations
def mutate_string(string, position, character):
    string= string[:position] + character + string[position+1:]
    return string
if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


#Find a String
def count_substring(string, sub_string):
    count=0
    for i in range(len(string)):
        if sub_string==string[i:i+len(sub_string)]:
            count+=1
    return count
if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)


#String Validators
if __name__ == '__main__':
    s = input()
    lowercase=0
    uppercase=0
    jj=0
    kk=0
    ll=0
    uu=0
    alpha=0
    for al in s:
        if al.isalnum():
            alpha+=1
    for j in s:
        if j.isalpha():
            jj+=1
    for k in s:
        if k.isdigit():
            kk+=1
    for l in s:
        if l.islower():
            ll+=1
    for u in s:
        if u.isupper():
            uu+=1
    if alpha!=0:
        print(True)
    else:
        print(False)
    if jj!=0:
        print (True)
    else:
        print(False)
    if kk!=0:
        print (True)
    else:
        print(False)
    if ll!=0:
        print (True)
    else:
        print(False)
    if uu!=0:
        print (True)
    else:
        print(False)


#Text Wrap
import textwrap
def wrap(string, max_width):
    wrapped=textwrap.fill(string,max_width)
    return wrapped
if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


#Designer Door Mat
if __name__=="__main__":
    n, m = map(int, input().split())
    for i in range(1, n, 2):
        print((i * ".|.").center(m, "-"))
    print("WELCOME".center(m, "-"))
    for i in range(n - 2, 0, -2):
        print((i * ".|.").center(m, "-"))


#Capitalize!
import math
import os
import random
import re
import sys
def solve(s):
    lis=s.split(" ")
    result_string=""
    i=0
    for l in range(len(lis)):
        lis[l]=lis[l].capitalize()
    result_string=" ".join(lis)
    return result_string
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = solve(s)
    fptr.write(result + '\n')
    fptr.close()


#The Minion Game
def minion_game(string):
    totalscoreS = 0
    totalscoreK = 0
    for i in range(len(string)):
        if string[i] not in 'AEIOU':
            totalscoreS += len(string) - i
        else:
            totalscoreK += len(string) - i
    if totalscoreS > totalscoreK:
        print("Stuart", totalscoreS)
    elif totalscoreS < totalscoreK:
        print("Kevin", totalscoreK)
    else:
        print("Draw")
if __name__ == '__main__':
    s = input()
    minion_game(s)


#Merge The Tools!
def merge_the_tools(string, k):
    parts=[string[i:i+k] for i in range(0,len(string),k)]
    uniques=[]
    for part in parts:
        uniques.append(''.join(sorted(set(part),key=part.index)))
    for i in uniques:
        print(i)
if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)




#Sets

#Introduction To Sets
def average(array):
    # your code goes here
    A=set(array)
    return sum(A)/len(A)
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)


#No Idea!
if __name__ == '__main__':
    import collections
    n,m = input().split()
    array= input().split()
    A=set(input().split())
    B=set(input().split())
    happiness=0
    for i in array:
        if i in A:
            happiness+=1
        if i in B:
            happiness-=1
    print (happiness)


#Symmetric Difference
if __name__ == '__main__':
    m = int(input())
    a = set(input().split())
    n = int(input())
    b = set(input().split())    
    s = list(map(int,a.union(b) - a.intersection(b)))
    s.sort()
    for i in s:
        print(i)


#Set.add()
if __name__=="__main__":
    n=int(input())
    countries=set()
    for i in range(n):
        countries.add(input())
    print (len(countries))


#Set.Union()
if __name__=="__main__":
    n=int(input())
    engnews=set(map(int,input().split()))
    b=int(input())
    frnews=set(map(int,input().split()))
    merged=engnews.union(frnews)
    print (len(merged))


#Set.Intersection()
if __name__=="__main__":
    n=int(input())
    engnews=set(map(int,input().split()))
    b=int(input())
    frnews=set(map(int,input().split()))
    merged=engnews.intersection(frnews)
    print (len(merged))


#Set.difference()
if __name__=="__main__":
    n=int(input())
    engnews=set(map(int,input().split()))
    b=int(input())
    frnews=set(map(int,input().split()))
    diff=engnews.difference(frnews)
    print (len(diff))


#Set.Symmetric_difference() 
if __name__=="__main__":
    n=int(input())
    engnews=set(map(int,input().split()))
    b=int(input())
    frnews=set(map(int,input().split()))
    symm=engnews.symmetric_difference(frnews)
    print (len(symm))




#Collections

#collections.counter()
if __name__ == '__main__':
    from collections import Counter
    x=int(input())
    size=[int(i) for i in list((input().split()))]
    cust=int(input())
    orders = [[int(i) for i in input().split()]
        for i in range(cust)]
    stock = Counter(size)
    profit = 0
    for order in orders:
        shoe = order[0]
        price = order[1]
        if shoe in stock and stock[shoe] > 0:
            stock[shoe] -= 1 
            profit += price
    print(profit)


#Collections.namedtuple()
if __name__=="__main__":
    n = int(input())
    curr=input().split().index('MARKS')
    marks = [input().split()[curr] for i in range(n)]
    print(round((sum(list(map(int, marks)))/n),2))




#Date and Time

#Calendar Module
if __name__=="__main__":
    import calendar
    mm,dd,yy=map(int,input().split())
    day=calendar.weekday(yy,mm,dd)
    days=["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"]
    print(days[day])




#Regex and Parsing challenges

#Detect Floating Point Number
if __name__=="__main__":
    import re
    pattern=r'^[+-]?[0-9]*\.[0-9]+$'
    for _ in range(int(input())):
        if re.search(pattern, input()):
            print('True')
        else:
            print('False')


#Re.split()
if __name__=="__main__":
    regex_pattern = r"[,.]"
    import re
    print("\n".join(re.split(regex_pattern, input())))


#Group(), Groups() & Groupdict()
if __name__=="__main__":
    import re
    s=input()
    search=re.search(r"([a-zA-Z0-9])\1", s)
    if search:
        print(search.group(1))
    else:
        print(-1)


#Re.findall() & Re.finditer()
if __name__=="__main__":
    import re
    control=re.findall(r'(?<=[qwrtypsdfghjklzxcvbnm])[AEIOUaeiou]{2,}(?=[qwrtypsdfghjklzxcvbnm])', input())
    if control:
        for i in control:
            print(i)
    else:
        print("-1")


#Re.start() & Re.end() 
if __name__=="__main__":
    import re
    S = str(input())
    k = str(input())
    ind = tuple(re.finditer(f"(?=({k}))", S))
    if re.search(k, S):
        for i in ind:
            print((i.start(1), i.end(1)-1))
    else:
        print((-1, -1))


#Regex Substitution
if __name__=="__main__":
    import re
    n = int(input())
    mod=""
    for i in range(n) :
        mod = str(input())
        mod = re.sub(r"(?<=\s)&&(?=\s)","and",mod)
        mod = re.sub(r"(?<=\s)\|\|(?=\s)", "or",mod)
        print(mod)


#Validating Phone Number 
if __name__=="__main__":
    import re
    N=int(input())
    for i in range(N):
        string = input()
        v = bool(re.search('^[789]',string))
        if v==True and len(string)==10 and string.isdigit():
            print("YES")
        else:
            print("NO")


#Validating and Parsing Email Address
if __name__=="__main__":
    import email.utils
    import re
    def valid_email(email):
        pattern = r'^[a-zA-Z][a-zA-Z0-9._-]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$'
        return bool(re.match(pattern, email))
    n = int(input())
    for _ in range(n):
        line = input()
        name, mail = email.utils.parseaddr(line)
        if valid_email(mail):
            print(line)




#XML

#XML 1 - Find the Score
import sys
import xml.etree.ElementTree as etree
def get_attr_number(node):
    # your code goes here
    score=len(node.attrib)
    for i in node:
        score+=get_attr_number(i)
    return score
if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))


#XML2 - Find the Maximum Depth
import xml.etree.ElementTree as etree
maxdepth = 0
def depth(elem, level):
    global maxdepth
    # your code goes here
    maxdepth = max(maxdepth,level+1)
    [depth(a, level + 1) for a in elem]
    return maxdepth
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)




#Numpy

#Arrays
import numpy
def arrays(arr):
    rev = numpy.array(arr[::-1], float)
    return rev
arr = input().strip().split(' ')
result = arrays(arr)
print(result)


#Shape and Reshape
import numpy
if __name__=='__main__':
    lis= list(map(int, input().split()))
    nump=numpy.array(lis)
    nump.shape=(3,3)
    print (nump)


#Transpose and Flatten
import numpy
if __name__ =="__main__": 
    N,M=list(map(int, input().split()))
    arr=[]
    for i in range(N):
        arr.append(list(map(int, input().split())))
    num=numpy.array(arr)
    tran=numpy.transpose(num)
    print (tran)
    print (num.flatten())


#Concatenate
if __name__=="__main__":   
    import numpy as np
    N,M,P=map(int,input().split())
    for i in range(N):
        arr1 = [list(map(int, input().split()))] 
    for i in range(M):
        arr2 = [list(map(int, input().split()))]
    array1 = np.array(arr1)
    array2= np.array(arr2)
    concat=np.concatenate((array1,array2),axis=0)
    mylist=list(concat)
    x=[]
    for i in range(N):
        x.append(mylist[0])
    for j in range(M):
        x.append(mylist[1])
    print (np.array(x))


#Zeros and Ones
if __name__=="__main__":
    import numpy as np
    lis=list(map(int, input().split()))
    print(np.zeros(lis, dtype=int))
    print(np.ones(lis, dtype=int))  


#Eye and Identity
if __name__=="__main__":    
    import numpy as np
    N,M=map(int,input().split())
    np.set_printoptions(legacy="1.13")
    print (np.eye(N,M))


#Array Mathematics
if __name__=="__main__":
    import numpy 
    n,m=list(map(int,input().split()))
    a = numpy.array([input().split() for _ in range(n)],int)
    b = numpy.array([input().split() for _ in range(n)],int)
    print(numpy.add(a,b))
    print(numpy.subtract(a,b))
    print(numpy.multiply(a,b))
    print(numpy.floor_divide(a,b))
    print(numpy.mod(a,b))
    print(numpy.power(a,b))


#Floor, Ceil and Rint
if __name__== "__main__":    
    import numpy as np
    a=np.array(list(map(float,input().split())))
    np.set_printoptions(legacy='1.13')
    print (np.floor(a))
    print (np.ceil(a))
    print (np.rint(a))


#Sum and Prod
if __name__=="__main__":    
    import numpy as np
    n,m= map(int,input().split())
    arr=[]
    for i in range(n):
        arr.append(input().split())
    myarray=np.array(arr,int)
    temp=np.sum(myarray,axis=0)
    print(np.prod(temp))  


#Min and Max
if __name__=="__main__":
    import numpy as np
    n,m=map(int,input().split())
    mylis=[]
    for i in range(n):
        lis=list(map(int,input().split()))
        mylis.append(lis)
    arr=np.array(mylis,int)
    print(max(np.min(arr,axis=1)))


#Mean, Var, and Std
if __name__=="__main__":
    import numpy as np
    n,m=map(int,input().split())
    mylis=[]
    for i in range(n):
        lis=list(map(int,input().split()))
        mylis.append(lis)
    arr=np.array(mylis,int)
    print(np.mean(arr,axis=1))
    print(np.var(arr,axis=0))
    print(round(np.std(arr,axis=None),11))


#Dot and Cross
if __name__=="__main__":
    import numpy as np
    n=int(input())
    a = np.array([input().split() for i in range(n)],int)
    b = np.array([input().split() for i in range(n)],int)
    print (np.dot(a,b))


#Inner and Outer
if __name__=="__main__":
    import numpy as np
    a=np.array(input().split(),int)
    b=np.array(input().split(),int)
    print (np.inner(a,b))
    print (np.outer(a,b))


#Polynomials
if __name__=="__main__":
    import numpy as np
    coeff=list(map(float,input().split()))
    x=int(input())
    print(np.polyval(coeff,x))


#Linear Algebra
if __name__=="__main__":
    import numpy as np
    n=int(input())
    a=np.array([input().split() for i in range(n)],float)
    print (round(np.linalg.det(a),2))




#Problem2

#Birthday Cake Candles
import math
import os
import random
import re
import sys
def birthdayCakeCandles(candles):
    candles=list(candles)
    maximum=max(candles)
    count=0
    for i in candles:
        if i==maximum:
            count+=1
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


#Number Line Jumps
import math
import os
import random
import re
import sys
def kangaroo(x1, v1, x2, v2):
    # Write your code here
    if v1<=v2 and x2>=x1:
        return "NO"
    else:
        if (x2 - x1) % (v1 - v2) == 0:
            return "YES"
        else:
            return "NO"
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


#Viral Advertising
import math
import os
import random
import re
import sys
def viralAdvertising(n):
    # Write your code here
    days=[i for i in range(n)]
    shared=[]
    liked=[]
    comulative=[]
    shared.append(5)
    liked.append(2)
    comulative.append(2)   
    for i in range(1,n):
            shared.append(liked[i-1]*3)
            liked.append(shared[i]//2)
            comulative.append(liked[i]+comulative[i-1])
    return comulative[n-1]
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


#Recursive Digit Sum
import math
import os
import random
import re
import sys
#It works only for 8 tests on 11. It doesn't work for a large lenght of string n (where the input says {truncated}), but it works for a large number k.
def somma(string):
    number=0
    for char in string:
        number+=int(char)
    return number
def superDigit(n, k):
    # Write your code here
    string=""
    for i in range(k):
        string=string+n
    num1=string[0:int(len(string)/2)]
    num2=string[int(len(string)/2):]
    while True:
        if len(num1)>1:
            num1=str(somma(num1))
        else:
            break   
    while True:
        if len(num2)>1:
            num2=str(somma(num2))
        else:
            break     
    if(int(num1)+int(num2))>=10:
        return int(str(int(num1)+int(num2))[0]) + int(str(int(num1)+int(num2))[1])
    else:
        return int(num1)+int(num2)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


#Insertion Sort - Part 1
import math
import os
import random
import re
import sys
def insertionSort1(n, arr):
    # Write your code here
    elem=arr[n-1]
    i=n-2
    while(elem<arr[i] and i>=0):
        arr[i+1]=arr[i]
        print(*arr)
        i=i-1
    arr[i+1]=elem
    print(*arr)    
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


#Insertion Sort - Part 2
import math
import os
import random
import re
import sys
def insertionSort2(n, arr):
    # Write your code here
    totarr=arr
    for i in range(1,len(totarr)):
        newarr=totarr[0:i+1]
        temp=totarr[i+1:]
        newarr.sort()
        totarr=newarr+temp
        newstr=""
        for k in totarr:
            newstr=newstr+str(k)+" "
        print(newstr)







