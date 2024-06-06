from tkinter import *
import tkinter as tk
import cv2
from tkinter import filedialog
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile                            
import imutils
from PIL import Image
from PIL import ImageTk
from sklearn.model_selection import KFold
import time
from PIL import ImageTk, Image
from skimage.filters import median
from numpy import load
from numpy import save
from skimage.color import rgb2gray
from tkinter import messagebox
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
from skimage import morphology
from PIL import Image, ImageTk




class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.pack(fill=BOTH, expand=1)  # Fill the entire window

        # Load the background image
        image_path = "bg.jpg"  # Replace this with the path to your image
        img = Image.open(image_path)
        img = img.resize((1400, 1400))  # Resize the image to fit the window size
        self.background_image = ImageTk.PhotoImage(img)

        # Create a label with the background image
        self.background_label = Label(self, image=self.background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Place the label to cover the entire window

        # Changing the title of our master widget      
        self.master.title("Window with Background Image")

        
        
                 
        
        w = tk.Label(self.master, 
		 text="skin prick detection using convolutional neural network",
		 fg = "light green",
		 bg = "brown",
		 font = "Helvetica 20 bold italic")
        w.pack()
        w.place(x=350, y=0)
        # creating a button instance
        quitButton = Button(self,command=self.query,text="browse input",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=350, y=475)

        
        load = Image.open("logo.jfif")
        render = ImageTk.PhotoImage(load)

        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=224, width=224, bg='white')
        image1.image = render
        image1.place(x=300, y=200)
        
        global T


    def query(self, event=None):
        global T,rep
        rep = filedialog.askopenfilenames()
        
        img = cv2.imread(rep[0])
        
        img = cv2.resize(img,(250,250))
        
        Input_img=img.copy()
        print(rep[0])
        self.from_array = Image.fromarray(cv2.resize(img,(250,250)))
        load = Image.open(rep[0])
        render = ImageTk.PhotoImage(load.resize((250,250)))
       
        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=224, width=224, bg='white')
        image1.image = render
        image1.place(x=300, y=200)

        
        clas1 = [item[10:-1] for item in sorted(glob("./dataset/*/"))]
        from keras.preprocessing import image
        from tqdm import tqdm

        def path_to_tensor(img_path, width=224, height=224):
            print(img_path)
            img = image.load_img(img_path, target_size=(width, height))
            print(img.size)
            x = image.img_to_array(img)
            return np.expand_dims(x, axis=0)

        def paths_to_tensor(img_paths, width=224, height=224):
            list_of_tensors = [path_to_tensor(img_paths, width, height)]
            return np.vstack(list_of_tensors)

        from tensorflow.keras.models import load_model
        model = load_model('trained_model_DNN1.h5')
        main_img = cv2.imread(rep[0])
        test_tensors = paths_to_tensor(rep[0]) / 255
        pred = model.predict(test_tensors)
        x = np.argmax(pred)
        predicted_folder = clas1[x]



            

        Disease_info=""
            
        if predicted_folder == 'Actinickeratosis':
            
            Disease_info = "Precaution: Actinic keratosis (AK) can progress to skin cancer. Regular skin checks and sun protection are crucial for early detection and prevention."
            
                          
        elif predicted_folder == 'AtopicDermatitis':
            Disease_info = "Precaution: To avoid harsh soaps and detergents, and opt for fragrance-free moisturizers to prevent flare-ups and soothe skin irritation. Additionally, minimize exposure to triggers like extreme temperatures and certain fabrics to help manage symptoms effectively."
            

        elif predicted_folder == 'Benignkeratosis':
            Disease_info = "Precaution: Avoid prolonged sun exposure and always wear sunscreen with high SPF to reduce the risk of benign keratosis development. Regularly examine your skin for any changes or new growths and consult a dermatologist if you notice anything suspicious."
            

        elif predicted_folder == 'Dermatofibroma':
            Disease_info = "Precaution: Avoid picking or scratching at the dermatofibroma to prevent irritation and potential infection. Regularly monitor any changes in size, shape, or color and consult a dermatologist if concerned."
            


        elif predicted_folder == 'Melanocyticnevus':
            Disease_info = "Precaution: Protect your nevus from excessive sun exposure by using sunscreen, wearing protective clothing, and seeking shade when outdoors to reduce the risk of potential complications."
            


        elif predicted_folder == 'Melanoma':
            Disease_info = "Precaution: Regularly check your skin for any new or changing moles or spots, and seek prompt medical attention for any suspicious changes. Always wear sunscreen with a high SPF, protective clothing, and seek shade during peak sun hours to reduce UV exposure."


        elif predicted_folder == 'Squamouscellcarcinoma':
            
            Disease_info = "Precaution: Protect your skin from prolonged sun exposure with clothing and sunscreen to lower your risk of developing squamous cell carcinoma. Regularly inspect your skin for any new or changing lesions, and consult a dermatologist if you notice anything unusual."
            

        elif predicted_folder == 'TineaRingwormCandidiasis':
            Disease_info = " Precaution: Keep affected areas clean and dry to prevent the spread of Tinea, Ringworm, and Candidiasis."

        elif predicted_folder == 'Vascularlesion':
            Disease_info = "Precaution: To prevent complications from vascular lesions, maintain a healthy lifestyle with regular exercise and a balanced diet, and promptly seek medical attention for any changes in size, color, or symptoms of the lesion. "
            

    
        print('Given image is  = ' + clas1[x])
        res = 'predicted skin prick is ' + clas1[x]
##        pr= predicted_folder

        T = Text(self, height=8, width=65)
        T.place(x=650, y=250)
        T.insert(END, res)
        #T.insert(END, '\n',Disease_info)
        # Display information about the vitamin
        T.insert(END, "\n\n" +Disease_info )

        # Show messagebox based on the prediction
        if x == clas1.index('normal'):
            messagebox.showinfo("Result", "Patient is normal")
        else:
            messagebox.showinfo("Result", "Patient has emergency")




from tkinter import messagebox
from PIL import Image, ImageTk

class LoginWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.config(bg='white')
        self.master.title("Login")

        # Load background image
        bg_image = Image.open("log.png")
        bg=bg_image.resize((1400,1400))
        bg_render = ImageTk.PhotoImage(bg)

        # Create a label to hold the background image
        self.background_label = Label(self, image=bg_render)
        self.background_label.image = bg_render
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.pack(fill=BOTH, expand=1)
        
        w = Label(self, text="Login Page", fg="#e6f2ff", bg="black", font="Helvetica 20 bold italic")
        w.pack()
        w.place(x=650, y=50)
        
        self.username_label = Label(self, text="Username:")
        self.username_label.place(x=550, y=100)
        
        self.username_entry = Entry(self)
        self.username_entry.place(x=650, y=100)
        
        self.password_label = Label(self, text="Password:")
        self.password_label.place(x=550, y=150)
        
        self.password_entry = Entry(self, show="*")
        self.password_entry.place(x=650, y=150)
        
        self.login_button = Button(self, text="Login", command=self.login)
        self.login_button.place(x=650, y=200)
    
    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        # Add your login logic here
        # For simplicity, let's just assume username is "admin" and password is "password"
        if username == "jai" and password == "chandru":
            self.master.switch_frame(Window)

        elif username == "Aishwarya" and password == "29":
             self.master.switch_frame(Window)

        elif username == "Chithra" and password == "2522":
             self.master.switch_frame(Window)

        elif username == "Amreen" and password == "1234":
             self.master.switch_frame(Window)    






             
             
        else:
            messagebox.showerror("Error", "Invalid username or password")    
             
        


class MainApplication(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        
        self.title("skin prick detection")
        self.geometry("1400x720")
        self.current_frame = None
        self.switch_frame(LoginWindow)
        
    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = new_frame
        self.current_frame.pack()

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()



    
