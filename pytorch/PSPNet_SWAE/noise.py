        ###### Yurong Chen 
        
        '''
        experiment 1: AddElementwise 
        '''     
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ## Noise image
        aug = iaa.AddElementwise((-40, 40))
        img = aug(images=img)
        ## 
        img = Image.fromarray(img)
        
        '''
        experiment 2: AdditiveGaussianNoise 
        '''   
        aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)) # 0.2 decide the degree of noise
        
        '''
        experiment 3: AdditiveLaplaceNoise 
        '''           
        aug = iaa.AdditiveLaplaceNoise(scale=(0, 0.3*255))
        
        '''
        experiment 4: AdditivePoissonNoise 
        '''           
        aug = iaa.AdditivePoissonNoise(40) # 40 decide the variables

        '''
        experiment 5: MultiplyElementwise 
        '''           
        aug = iaa.MultiplyElementwise((0.5, 1.5))
 
        '''
        experiment 6: Dropout 
        '''          
        aug = iaa.Dropout(p=(0, 0.2))  
 
        '''
        experiment 7: ImpulseNoise 
        '''         
        aug = iaa.ImpulseNoise(0.1) # Replace 10% of all pixels with impulse noise: 
   
        '''
        experiment 8: SaltAndPepper 
        '''         
        aug = iaa.SaltAndPepper(0.1)     
        
        '''
        experiment 9: Salt
        '''         
        aug = iaa.Salt(0.1) #Pepper(0.1)
        
        '''
        experiment 9: JpegCompression
        '''           
        aug = iaa.JpegCompression(compression=(70, 99))
        
        
        '''
        experiment 9: GaussianBlur
        '''         
        aug = iaa.GaussianBlur(sigma=(0.0, 10.0))
        aug = iaa.AverageBlur(k=(2, 11))
        
        

