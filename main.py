import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
import tkinter.messagebox
import os

main_window=Tk()
main_window.title('Monocular Depth Estimation')
main_window.geometry("1000x700")
main_window.configure(bg='lightyellow')

text1='Monocular Depth Estimation'
lb1 = tk.Label(main_window, text=text1, width=30,height=3,
               font=('Times',30), fg='black',  bg='olive')

lb1.place(x=260,y=30)

text2='康敏桐 3180103128'
lb1 = tk.Label(main_window, text=text2, width=20,height=2,
               font=('Times',20), fg='black',  bg='LightSkyBlue')

lb1.place(x=370,y=160)

text3='Model Path'
lb1 = tk.Label(main_window, text=text3, width=20,height=2,
               font=('Times',20), fg='black',  bg='Cyan')
lb1.place(x=60,y=260)

text4='Input Image Path'
lb2 = tk.Label(main_window, text=text4, width=20,height=2,
               font=('Times',20), fg='black',  bg='Cyan')
lb2.place(x=60,y=360)

text5='Output Image Path'
lb3 = tk.Label(main_window, text=text5, width=20,height=2,
               font=('Times',20), fg='black',  bg='Cyan')
lb3.place(x=60,y=460)

text_show1=Text(main_window, height=2, width=46, font=('Times',20))
text_show1.place(x=300,y=260)
text_show1.insert('0.0','Please choose model!')

def choose_file1():
    file_path = askopenfilename(title='choose file', initialdir=r'./',
                               filetypes=[('.model','.tar')],
                               #defaultextension='.pth',
                               #initialfile='ys3.jpg'
                               )
    if file_path!='':
        text_show1.delete('1.0', 'end')
        text_show1.insert('0.0', file_path)
button2 = tk.Button(text="choose file", command=choose_file1, font=('Times',30), fg='black',  bg='yellow')
button2.place(x=800, y=260)

text_show2=Text(main_window, height=2, width=46, font=('Times',20))
text_show2.place(x=300,y=360)
text_show2.insert('0.0','Please choose input path!')

def choose_file2():
    file_path = askopenfilename(title='choose file', initialdir=r'./',
                               filetypes=[('.pic','.png')],
                               #defaultextension='.pth',
                               #initialfile='ys3.jpg'
                               )
    if file_path!='':
        text_show2.delete('1.0', 'end')
        text_show2.insert('0.0', file_path)
button3 = tk.Button(text="choose file", command=choose_file2, font=('Times',30), fg='black',  bg='yellow')
button3.place(x=800, y=360)

text_show3=Text(main_window, height=2, width=46, font=('Times',20))
text_show3.place(x=300,y=460)
text_show3.insert('0.0','Please choose output directory!')

def choose_file3():
    file_path = askdirectory(title='choose DIRECTORY', initialdir=r'./',
                               #filetypes=[('.model','.pth')],
                               #defaultextension='.pth',
                               #initialfile='ys3.jpg'
                               )
    if file_path!='':
        text_show3.delete('1.0', 'end')
        text_show3.insert('0.0', file_path)
button3 = tk.Button(text="choose dir", command=choose_file3, font=('Times',30), fg='black',  bg='yellow')
button3.place(x=800, y=460)



def EvaluateModel():
    model_path=str(text_show1.get('1.0','end')).replace('\n','')
    input_path=str(text_show2.get('1.0','end')).replace('\n','')
    output_dir=str(text_show3.get('1.0','end')).replace('\n','')


    if model_path.endswith('.tar') and (input_path.endswith('.png') or input_path.endswith('.PNG')) and os.path.exists(output_dir):
        from evaluate_single_img import evaluate_single_img
        evaluate_single_img(model_path, input_path, output_dir)
        tkinter.messagebox.showinfo(message='Output has been saved at {}'.format(output_dir))
    else:
        tkinter.messagebox.showerror(message='Please check the model path, input path and output directory!')


button1 = tk.Button(text="Test Model", command=EvaluateModel, font=('Times',30), fg='black',  bg='yellow')
button1.place(x=380, y=570)


main_window.mainloop()