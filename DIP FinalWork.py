import tkinter as tk
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import cv2
import numpy as np
from tkinter import filedialog


class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # 图像可选择
        self.image_path_var = tk.StringVar()
        self.image_path_var.set("请输入图像路径")
        self.image_path_entry = tk.Entry(self, textvariable=self.image_path_var)
        self.image_path_entry.grid(row=0, column=0, padx=20, pady=20)

        self.browse_button = tk.Button(self)
        self.browse_button["text"] = "Browse Image"
        self.browse_button["command"] = self.browse_image
        self.browse_button.grid(row=0, column=1, padx=20, pady=20) 
        # 设置第0列的最小宽度
        self.columnconfigure(0, minsize=5)

        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.grid(row=1, column=0, padx=20, pady=20) 

        self.spatial_filtering_button = tk.Button(self)
        self.spatial_filtering_button["text"] = "Spatial Filtering"
        self.spatial_filtering_button["command"] = self.spatial_filtering
        self.spatial_filtering_button.grid(row=2, column=0, padx=20, pady=20) 
        
        self.kernel_label = tk.Label(self,text='Kernel:')
        self.kernel_label.grid(row=2, column=1, padx=1, pady=1 ,sticky="e")
        self.kernel_var = tk.StringVar()
        self.kernel_var.set("0 -1 0\n-1 5 -1\n0 -1 0")
        self.kernel_entry = tk.Text(self, height=3, width=15)
        self.kernel_entry.insert(tk.END, self.kernel_var.get())
        self.kernel_entry.grid(row=2, column=2, padx=1, pady=1)

        self.frequency_filter_button = tk.Button(self)
        self.frequency_filter_button["text"] = "Frequency Filtering"
        self.frequency_filter_button["command"] = self.frequency_filter
        self.frequency_filter_button.grid(row=3, column=0, padx=20, pady=20) 

        self.filter_type_label = tk.Label(self , text= 'lowpassORhighpass:')
        self.filter_type_label.grid(row=3, column=1, padx=1, pady=1 ,sticky="e")
        self.filter_type_entry = tk.Entry(self)
        self.filter_type_entry.grid(row=3, column=2, padx=20, pady=20)

        self.meanshift_button = tk.Button(self)
        self.meanshift_button["text"] = "meanshift"
        self.meanshift_button["command"] = self.meanshift
        self.meanshift_button.grid(row=4, column=0, padx=20, pady=20) 

        self.low_threshold1_label = tk.Label(self, text="low_threshold:")
        self.low_threshold1_label.grid(row=4, column=1, padx=1, pady=1,sticky="e")
        self.low_threshold1_entry = tk.Entry(self)
        self.low_threshold1_entry.grid(row=4, column=2, padx=1, pady=1)

        self.high_threshold1_label = tk.Label(self, text="high_threshold:")
        self.high_threshold1_label.grid(row=4, column=3, padx=1, pady=10)
        self.high_threshold1_entry = tk.Entry(self)
        self.high_threshold1_entry.grid(row=4, column=4, padx=1, pady=10)

        self.median_filtering_button = tk.Button(self)
        self.median_filtering_button["text"] = "median_filtering"
        self.median_filtering_button["command"] = self.median_filtering
        self.median_filtering_button.grid(row=5, column=0, padx=20, pady=20) 

        self.noise_ratio_label = tk.Label(self, text="noise_ratio:")
        self.noise_ratio_label.grid(row=5, column=1, padx=1, pady=1,sticky="e")
        self.noise_ratio_entry = tk.Entry(self)
        self.noise_ratio_entry.grid(row=5, column=2, padx=1, pady=1)

        self.canny_button = tk.Button(self)
        self.canny_button["text"] = "canny"
        self.canny_button["command"] = self.canny
        self.canny_button.grid(row=6, column=0, padx=20, pady=20) 

        self.low_threshold_label = tk.Label(self, text="low_threshold:")
        self.low_threshold_label.grid(row=6, column=1, padx=1, pady=1,sticky="e")
        self.low_threshold_entry = tk.Entry(self)
        self.low_threshold_entry.grid(row=6, column=2, padx=1, pady=1)

        self.high_threshold_label = tk.Label(self, text="high_threshold:")
        self.high_threshold_label.grid(row=6, column=3, padx=1, pady=10)
        self.high_threshold_entry = tk.Entry(self)
        self.high_threshold_entry.grid(row=6, column=4, padx=1, pady=10)

        self.canny_button = tk.Button(self)
        self.canny_button["text"] = "histogram"
        self.canny_button["command"] = self.histogram
        self.canny_button.grid(row=7, column=0, padx=20, pady=20)

        self.reference_var = tk.StringVar()
        self.reference_var.set("请输入图像路径")
        self.reference_entry = tk.Entry(self, textvariable=self.reference_var)
        self.reference_entry.grid(row=7, column=1, padx=1, pady=1,sticky="e")

        self.browse1_button = tk.Button(self)
        self.browse1_button["text"] = "reference Image"
        self.browse1_button["command"] = self.browse1_image
        self.browse1_button.grid(row=7, column=2, padx=1, pady=1) 

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.grid(row=8, column=1, padx=30, pady=30) 
        self.master.geometry("800x650")
    
    #空域滤波
    def spatial_filtering(self):  
        image_path = self.image_path_entry.get()
        image = cv2.imread(image_path)
        kernel_string = self.kernel_entry.get("1.0", tk.END)
        kernel_lines = kernel_string.strip().split("\n")
        kernel_values = []
        for line in kernel_lines:
            values = line.strip().split()
            kernel_values.append([float(value) for value in values])
        kernel = np.array(kernel_values)

        channels = image.shape[2]
        image_height, image_width, _ = image.shape
        kernel_height, kernel_width = kernel.shape
        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1
        output_image = np.zeros((output_height, output_width, channels), dtype=np.uint8)
        for h in range(output_height):
            for w in range(output_width):
                for c in range(channels):
                    output_image[h, w, c] = np.sum(image[h:h+kernel_height, w:w+kernel_width, c] * kernel)

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title('Convolution Result')
        plt.axis('off')

        plt.show()

    #频域滤波
    def frequency_filter(self, cutoff_freq=0.3):
        image_path = self.image_path_entry.get()
        filter_type = self.filter_type_entry.get()
        image = cv2.imread(image_path)
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        y_channel = yuv_image[:,:,0].astype(np.float32) / 255.0

        dct = cv2.dct(y_channel)

        mask = np.zeros_like(dct)
        if filter_type == 'lowpass':
            mask[:int(dct.shape[0]*cutoff_freq), :int(dct.shape[1]*cutoff_freq)] = 1
        elif filter_type == 'highpass':
            mask[int(dct.shape[0]*cutoff_freq):, int(dct.shape[1]*cutoff_freq):] = 1

        filtered_dct = dct * mask

        filtered_y_channel = cv2.idct(filtered_dct)

        filtered_yuv_image = yuv_image.copy()
        filtered_yuv_image[:,:,0] = np.clip(filtered_y_channel * 255.0, 0, 255)

        filtered_image = cv2.cvtColor(filtered_yuv_image, cv2.COLOR_YUV2BGR)

        cv2.imshow('Filtered Image', filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('filtered_image.jpg', filtered_image)    
    
    #meanshift图像分割
    def meanshift(self):
        image_path = self.image_path_entry.get()
        image = cv2.imread(image_path)
        low_threshold = int(self.low_threshold1_entry.get())
        high_threshold = int(self.high_threshold1_entry.get())
        shifted = cv2.pyrMeanShiftFiltering(image, low_threshold, high_threshold)#20 40
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('Original Image', image)
        cv2.imshow('Segmented Image', thresholded)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #median filtering
    def median_filtering(self):
        noise_ratio = self.noise_ratio_entry.get()
        image_path = self.image_path_entry.get()
        image = cv2.imread(image_path)
        A = image
        height, width, channels = image.shape
        num_noise_pixels = int(height * width * channels * noise_ratio)
        noisy_image = np.copy(image)

        for i in range(num_noise_pixels):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            c = np.random.randint(0, channels)
            if np.random.randint(0, 2) == 0:
                noisy_image[y, x, c] = 0
            else:
                noisy_image[y, x, c] = 255
        A_noise = noisy_image
        filtered_image = np.zeros((height, width, channels), dtype=np.uint8)
        padding_size = int((3 - 1) / 2)

        for i in range(padding_size, height - padding_size):
            for j in range(padding_size, width - padding_size):
                for c in range(channels):
                    sub_image = image[i - padding_size:i + padding_size + 1, j - padding_size:j + padding_size + 1, c]
                    # 对邻域像素排序，取中间值作为该像素的值
                    filtered_image[i, j, c] = np.median(np.sort(sub_image, axis=None))
        A_filtered = filtered_image
        # 显示和保存结果
        cv2.imshow("A", A)
        cv2.imshow("A_noise", A_noise)
        cv2.imshow("A_filtered", A_filtered)

    #canny
    def canny(self):
        image_path = self.image_path_entry.get()
        low_threshold = int(self.low_threshold_entry.get())
        high_threshold = int(self.high_threshold_entry.get())

        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)

        suppressed_image = np.zeros_like(gradient_magnitude)
        angle = gradient_direction * 180. / np.pi
        angle[angle < 0] += 180

        rows, cols = gradient_magnitude.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                q = 255
                r = 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j+1]
                    r = gradient_magnitude[i, j-1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_magnitude[i+1, j-1]
                    r = gradient_magnitude[i-1, j+1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_magnitude[i+1, j]
                    r = gradient_magnitude[i-1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_magnitude[i-1, j-1]
                    r = gradient_magnitude[i+1, j+1]

                if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                    suppressed_image[i, j] = gradient_magnitude[i, j]
                else:
                    suppressed_image[i, j] = 0

        edges = np.zeros_like(suppressed_image)
        strong_pixels = suppressed_image > high_threshold
        weak_pixels = (suppressed_image >= low_threshold) & (suppressed_image <= high_threshold)
        edges[strong_pixels] = 255
        for i in range(1, suppressed_image.shape[0]-1):
            for j in range(1, suppressed_image.shape[1]-1):
                if weak_pixels[i, j]:
                    if np.any(strong_pixels[i-1:i+2, j-1:j+2]):
                        edges[i, j] = 255

        cv2.imshow('orignal',image)
        cv2.imshow('Edges', edges)
        

    #histogram
    def histogram(self):
        image_path = self.image_path_entry.get()
        image = cv2.imread(image_path)
        reference_path = self.reference_entry.get()
        reference_image = cv2.imread(reference_path)
        histogram = np.zeros(256)
        cumulative_histogram = np.zeros(256)
        equalized_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                histogram[image[i, j]] += 1

        cumulative_histogram[0] = histogram[0]
        for i in range(1, 256):
            cumulative_histogram[i] = cumulative_histogram[i-1] + histogram[i]

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                equalized_image[i, j] = np.round((cumulative_histogram[image[i, j]] / cumulative_histogram.max()) * 255)

        histogram = np.zeros(256)
        cumulative_histogram = np.zeros(256)
        reference_equalized_image = np.zeros_like(reference_image)

        for i in range(reference_image.shape[0]):
            for j in range(reference_image.shape[1]):
                histogram[reference_image[i, j]] += 1

        cumulative_histogram[0] = histogram[0]
        for i in range(1, 256):
            cumulative_histogram[i] = cumulative_histogram[i-1] + histogram[i]

        for i in range(reference_image.shape[0]):
            for j in range(reference_image.shape[1]):
                reference_equalized_image[i, j] = np.round((cumulative_histogram[reference_image[i, j]] / cumulative_histogram.max()) * 255)

        matched_image = np.zeros_like(image)

        input_cumulative_histogram = np.zeros(256)
        for i in range(equalized_image.shape[0]):
            for j in range(equalized_image.shape[1]):
                input_cumulative_histogram[equalized_image[i, j]] += 1
        input_cumulative_histogram = np.cumsum(input_cumulative_histogram)

        reference_cumulative_histogram = np.zeros(256)
        for i in range(reference_equalized_image.shape[0]):
            for j in range(reference_equalized_image.shape[1]):
                reference_cumulative_histogram[reference_equalized_image[i, j]] += 1
        reference_cumulative_histogram = np.cumsum(reference_cumulative_histogram)

        # 进行规定化
        matched_lookup_table = np.zeros(256, dtype='uint8')
        for i in range(256):
            j = 0
            while reference_cumulative_histogram[j] < input_cumulative_histogram[i] and j < 255:
                j += 1
            matched_lookup_table[i] = j

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                matched_image[i, j] = np.array([matched_lookup_table[equalized_image[i, j][0]],
                                                matched_lookup_table[equalized_image[i, j][1]],
                                                matched_lookup_table[equalized_image[i, j][2]]])

        plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
        plt.subplot(122), plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)), plt.title('Equalized')
        plt.show()
        plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
        plt.subplot(122), plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)), plt.title('Matched')
        plt.show()

    def say_hi(self):
        print("hi there, this is my final DIPHomework!")
    
    def browse_image(self):
        # 打开文件对话框，让用户选择图像文件
        image_path = filedialog.askopenfilename()
        # 更新文本框显示的图像路径
        self.image_path_var.set(image_path)
    def browse1_image(self):
        # 打开文件对话框，让用户选择图像文件
        image_path = filedialog.askopenfilename()
        # 更新文本框显示的图像路径
        self.reference_var.set(image_path)


root = tk.Tk()
app = App(master=root)
app.mainloop()
