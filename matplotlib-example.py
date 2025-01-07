import matplotlib.pyplot as plt
x = [1,2,3,4,5]
y = [10,20,25,30,40]
plt.plot(x,y,marker='o', color='blue', label='Line plot')
plt.title("Example line plot")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.legend()
plt.show