import tkinter as tk
from tkinter import ttk,filedialog, messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from numpy.lib.format import read_magic


def linear_aprox(data):
    sx = sum(data[0])
    sxx = sum(map(lambda x: x**2, data[0]))
    sy = sum(data[1])
    sxy = sum([data[0][i]*data[1][i] for i in range(len(data[0]))])
    n = len(data[0])
    a = (sxy*n-sx*sy)/(sxx*n-sx*sx)
    b = (sxx*sy-sx*sxy)/(sxx*n-sx*sx)
    x_mean = sum(data[0])/len(data[0])
    y_mean = sum(data[1])/len(data[0])
    pirs = sum([(data[0][i]-x_mean)*(data[1][i]-y_mean) for i in range(n)])/(sum([(data[0][i]-x_mean)**2 for i in range(n)])*sum([(data[1][i]-y_mean)**2 for i in range(n)]))**0.5
    return [lambda x: a*x+b, f"Линейная функция: {a}x+{b}", f"Коэффициент корреляции Пирсона: {pirs}"]

def square_aprox(data):
    n = len(data[0])
    sx = sum(data[0])
    sxx = sum(map(lambda x: x**2, data[0]))
    sy = sum(data[1])
    sxy = sum([data[0][i]*data[1][i] for i in range(len(data[0]))])
    sxxx = sum(map(lambda x: x ** 3, data[0]))
    sxxxx = sum(map(lambda x: x ** 4, data[0]))
    sxxy = sum([data[0][i]**2 * data[1][i] for i in range(len(data[0]))])
    A = np.array([[n, sx, sxx], [sx, sxx, sxxx], [sxx, sxxx, sxxxx]])
    B = np.array([sy, sxy, sxxy])
    x = np.linalg.solve(A, B)
    return [lambda y: x[0]+y*x[1]+y**2*x[2],  f"Квадратичная функция: {x[0]}+{x[1]}*x+{x[2]}*x^2"]

def cube_aprox(data):
    n = len(data[0])
    sx = sum(data[0])
    sxx = sum(map(lambda x: x**2, data[0]))
    sy = sum(data[1])
    sxy = sum([data[0][i]*data[1][i] for i in range(len(data[0]))])
    sxxx = sum(map(lambda x: x ** 3, data[0]))
    sxxxx = sum(map(lambda x: x ** 4, data[0]))
    sxxxxx = sum(map(lambda x: x ** 5, data[0]))
    sxxxxxx = sum(map(lambda x: x ** 6, data[0]))
    sxxy = sum([data[0][i]**2 * data[1][i] for i in range(len(data[0]))])
    sxxxy = sum([data[0][i]**3 * data[1][i] for i in range(len(data[0]))])
    A = np.array([[n, sx, sxx, sxxx], [sx, sxx, sxxx, sxxxx], [sxx, sxxx, sxxxx, sxxxxx], [sxxx, sxxxx, sxxxxx, sxxxxxx]])
    B = np.array([sy, sxy, sxxy, sxxxy])
    x = np.linalg.solve(A, B)
    return [lambda y: x[0]+y*x[1]+y**2*x[2]+y**3*x[3], f"Кубическая функция: {x[0]}+{x[1]}*x+{x[2]}*x^2+{x[3]}*x^3"]

def exp_aprox(data):
    data2 = np.array([data[0], np.log(data[1])])
    sx = sum(data2[0])
    sxx = sum(map(lambda x: x**2, data2[0]))
    sy = sum(data2[1])
    sxy = sum([data2[0][i]*data2[1][i] for i in range(len(data2[0]))])
    n = len(data2[0])
    A = (sxy*n-sx*sy)/(sxx*n-sx*sx)
    B = (sxx*sy-sx*sxy)/(sxx*n-sx*sx)
    return [lambda x: np.exp(B)*np.exp(A*x), f"Экспоненцияальная функция: {np.exp(B)}*e^(x*{A})"]
def step_aprox(data):
    data2 = np.array([np.log(data[0]), np.log(data[1])])
    sx = sum(data2[0])
    sxx = sum(map(lambda x: x**2, data2[0]))
    sy = sum(data2[1])
    sxy = sum([data2[0][i]*data2[1][i] for i in range(len(data2[0]))])
    n = len(data2[0])
    A = (sxy*n-sx*sy)/(sxx*n-sx*sx)
    B = (sxx*sy-sx*sxy)/(sxx*n-sx*sx)
    return [lambda x: np.exp(B)*x**A, f"Степенная функция: {np.exp(B)}*x**{A})"]
def log_aprox(data):
    data2 = np.array([np.log(data[0]), data[1]])
    sx = sum(data2[0])
    sxx = sum(map(lambda x: x**2, data2[0]))
    sy = sum(data2[1])
    sxy = sum([data2[0][i]*data2[1][i] for i in range(len(data2[0]))])
    n = len(data2[0])
    A = (sxy*n-sx*sy)/(sxx*n-sx*sx)
    B = (sxx*sy-sx*sxy)/(sxx*n-sx*sx)
    return [lambda x: A*np.log(x)+B, f"Логарифмическая функция: {A}*ln(x)+{B})"]


def read_data():
    try:
        data = input_area.get("1.0", tk.END)
        data = data.strip().split('\n')
        d = [[],[]]
        for i in data:
            x, y = i.split()
            d[0].append(float(x.replace(',', '.')))
            d[1].append(float(y.replace(',', '.')))
        data = np.array(d)
        if data.shape[1]<3:
            delete_pre_res_func()
            messagebox.showerror("Ошибка!","Сообщение об ошибке: Ожидалось хотя бы 3 точки для построения аппроксимирующей функции")
            return 0
        return data
    except:
        delete_pre_res_func()
        messagebox.showerror("Ошибка!", "Сообщение об ошибке: Некорректные данные! Пожалуйста, введите данные в формате:\nx1 y1\nx2 y2")
        return 0

def draw_func():
    global data_for_save
    delete_pre_res_func()
    data = read_data()
    if type(data)==int:
        return 0
    ax.clear()
    draw_dots()
    fs=[]
    fs.append(["Линейная аппроксимация", linear_aprox(data)])
    fs.append(["Квадратичная аппроксимация", square_aprox(data)])
    fs.append(["Степенная аппроксимация", cube_aprox(data)])
    if data[0].all() >0:
        fs.append(["Логарифмическая аппроксимация", log_aprox(data)])
    if data[1].all() >0:
        fs.append(["Экспоненциальная аппроксимация", exp_aprox(data)])
    if data[1].all() >0 and data[0].all() >0:
        fs.append(["Степенная аппроксимация", step_aprox(data)])
    x = np.linspace(min(data[0]), max(data[0]), 1000)
    for i in range(len(fs)):
        funcc = fs[i]
        y = funcc[1][0](x)
        ax.plot(x, y, label=funcc[0])
        n = len(data[0])
        srkv = (sum([(data[1][i]-funcc[1][0](data[0][i]))**2 for i in range(len(data[0]))])/len(data[0]))**0.5
        fi_mean = sum([funcc[1][0](data[0][i]) for i in range(len(data[0]))])/len(data[0])
        r2 = 1-sum([(data[1][i]-funcc[1][0](data[0][i]))**2 for i in range(n)])/sum([(data[1][i]-fi_mean)**2 for i in range(n)])
        fs[i].append(srkv)
        fs[i].append(r2)
        if r2>=0.95:
            fs[i].append("высокой точности аппроксимации")
        elif r2>=0.75:
            fs[i].append("удовлетворительной аппроксимации")
        elif r2>=0.5:
            fs[i].append("слабой аппроксимации")
        else:
            fs[i].append("неточности аппроксимации")

    fs = sorted(fs, key=lambda x: x[2])
    root_label.config(text=config(data, fs[0]))
    for i in fs:
        data_for_save += config(data, i)
        data_for_save += '\n'
    ax.legend(loc='upper right', fontsize=7, framealpha=1, shadow=True)
    ax.grid( color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    #ax.axhline(0, color='black', linewidth=0.5)
    canvas.draw()
    return 0

def draw_dots():
    data = read_data()
    if type(data)!=int:
        ax.clear()
        ax.scatter(data[0], data[1], color='blue', marker='o', label='заданные точки')
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        #ax.axhline(0, color='black', linewidth=0.5)
        canvas.draw()
    return 0


def config(data, list1):
    return "Входные параметры:\nx:"+str(data[0])+ "\ny:"+str(data[1])+ "\nТип аппроксимирующей функции:" + str(list1[1][1]) + "\nЕе значения в заданных точках:"+str(list1[1][0](data[0])) + "\nСреднеквадратичное отклонение:" + str(list1[2]) + "\nКоэффициент детерминации:" + str(list1[3]) + ", можно говорить о " + str(list1[4]) + '\n'+ str(*list1[1][2:])

def delete_pre_res_func():
    global data_for_save
    Info_label.config(text="Информация о выполнении")
    root_label.config(text="")
    data_for_save=""


def show_frame(frame):
    frame.grid(row=0, column=0, sticky="nsew")

def save():
    if root_label.cget('text')=="":
        messagebox.showerror("Ошибка!",
                             f"Сообщение об ошибке: Перед сохранением рассчитайте аппроксимирующие функции, информацию о рассчете которых хотите сохранить")
        return 1
    filepath = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Текстовые файлы", "*.txt")]
    )
    if not filepath:
        return 1
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(data_for_save)
    messagebox.showinfo("Успех", f"Файл сохранён: {filepath}")
def read_from_file():
    try:
        with open(file_name.get(), 'r') as file:
            data = file.read()
            input_area.delete("1.0", tk.END)
            input_area.insert("1.0", data)
    except FileNotFoundError:
        messagebox.showerror("Ошибка!", f"Сообщение об ошибке: Файл '{file_name.get()}' не найден")
    except Exception as e:
        messagebox.showerror("Ошибка!", f"Неизвестная ошибка: {e}")



# cur_func=func1
# cur_func_name="sin(exp(x) + x)"
# cur_meth=left_pram_meth
# cur_meth_name="Метод левых прямоугольников"

data_for_save=""

root = tk.Tk()
root.title("Динамическое построение графиков")
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

int_frame = ttk.Frame(root)
int_frame.grid_rowconfigure(0, weight=1)
int_frame.grid_rowconfigure(1, weight=1)
int_frame.grid_rowconfigure(2, weight=1)
int_frame.grid_rowconfigure(3, weight=1)
int_frame.grid_rowconfigure(4, weight=1)
int_frame.grid_rowconfigure(5, weight=1)
int_frame.grid_rowconfigure(6, weight=1)
int_frame.grid_rowconfigure(7, weight=1)
int_frame.grid_rowconfigure(8, weight=1)
int_frame.grid_rowconfigure(9, weight=1)
int_frame.grid_rowconfigure(10, weight=1)
int_frame.grid_columnconfigure(0, weight=1)
int_frame.grid_columnconfigure(1, weight=1)

# nes_frame = ttk.Frame(root)
# nes_frame.grid_rowconfigure(0, weight=1)
# nes_frame.grid_rowconfigure(1, weight=1)
# nes_frame.grid_rowconfigure(2, weight=1)
# nes_frame.grid_rowconfigure(3, weight=1)
# nes_frame.grid_rowconfigure(4, weight=1)
# nes_frame.grid_rowconfigure(5, weight=1)
# nes_frame.grid_rowconfigure(6, weight=1)
# nes_frame.grid_rowconfigure(7, weight=1)
# nes_frame.grid_rowconfigure(8, weight=1)
# nes_frame.grid_rowconfigure(9, weight=1)
# nes_frame.grid_rowconfigure(10, weight=1)
# nes_frame.grid_rowconfigure(11, weight=1)
# nes_frame.grid_rowconfigure(12, weight=1)
# nes_frame.grid_rowconfigure(13, weight=1)
# nes_frame.grid_columnconfigure(0, weight=1)
# nes_frame.grid_columnconfigure(1, weight=1)

# # Поле для ввода погрешности
# accuracy_label = ttk.Label(int_frame, text="Точность:")
# accuracy_label.grid(row=0,column=0, padx=10, pady=10,sticky="ew")
# accuracy = ttk.Entry(int_frame)
# accuracy.insert(0, "0.01")
# accuracy.grid(row=0,column=1, padx=10, pady=10,sticky="ew")
#
# # Поле для ввода отрезка
# left_gran_label = ttk.Label(int_frame, text="левая граница:")
# left_gran_label.grid(row=1,column=0, padx=10, pady=10,sticky="ew")
# left_gran = ttk.Entry(int_frame)
# left_gran.insert(0, "0.0")
# left_gran.grid(row=1,column=1, padx=10, pady=10,sticky="ew")
#
#
# right_gran_label = ttk.Label(int_frame, text="правая граница:")
# right_gran_label.grid(row=2,column=0, padx=10, pady=10,sticky="ew")
# right_gran = ttk.Entry(int_frame)
# right_gran.insert(0, "1.0")
# right_gran.grid(row=2,column=1, padx=10, pady=10,sticky="ew")
# functions = ["sin(exp(x) + x)", "x ** sin(x) - 0.5 * x", "atan(x) * sin(x) * 9", "x ** 3 - 9 * x ** 2 + x + 11", "-3*x**3 - 5*x**2 + 4*x - 2", "1 / abs(x) ** 0.5", "1 / (1 - x)"]
# func_combobox = ttk.Combobox(int_frame, values=functions)
# func_combobox.set("sin(exp(x) + x)")
# #func_combobox.pack(pady=10)
# func_combobox.bind("<<ComboboxSelected>>", on_func_combobox_change)
# func_combobox.grid(row=3,column=0)
input_area_label = ttk.Label(int_frame, text="Точки аппроксимации:")
input_area_label.grid(row=1,column=0, columnspan=2)
input_area = tk.Text(int_frame)
input_area.grid(row=2, column=0, columnspan=2)


fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=int_frame)
draw_button = ttk.Button(int_frame, text="Нарисовать множество точек", command=draw_dots)
draw_button.grid(row=3, column=0, padx=10, pady=10,sticky="ew")
draw_button = ttk.Button(int_frame, text="Рассчитать аппроксимирующие функции", command=draw_func)
draw_button.grid(row=3, column=1, padx=10, pady=10,sticky="ew")

save_button = ttk.Button(int_frame, text="Сохранить результат", command=save)
save_button.grid(row=4, column=0, columnspan=2)
canvas.get_tk_widget().grid(row=5, column=0, columnspan=2)

Info_label = ttk.Label(int_frame, text="Информация о выполнении:")
Info_label.grid(row=6, column=0, columnspan=2)


root_label = ttk.Label(int_frame, text="")
root_label.grid(row=7, column=0, columnspan=2)


file_name_label = ttk.Label(int_frame, text="Название файла:")
file_name_label.grid(row=9, column=0, padx=10, pady=10, sticky="ew")
file_name = ttk.Entry(int_frame)
file_name.insert(0, "a.txt")
file_name.grid(row=9, column=1, padx=10, pady=10, sticky="ew")

load_f_button = ttk.Button(int_frame, text="Загрузить из файла", command=read_from_file)
load_f_button.grid(row=10, column=0, columnspan=2)

show_frame(int_frame)
# Запуск основного цикла
root.mainloop()
