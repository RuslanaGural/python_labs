#Створення прикладних програм на мові Python. Лабораторна робота №6.3. Гураль Руслана. FI-9119


print('''' Створення прикладних програм на мові Python. Лабораторна робота №6.3
  Гураль Руслана. FI-9119''')

st = "The only thing in life achieved without effort is failure"

startIndex = st.find('h')
endIndex = st.rfind('h')

st = st[startIndex+1:endIndex]
st = st[::-1]
stList = list(st)

print('st ---  ',st)
print('stList ---  ', stList)



