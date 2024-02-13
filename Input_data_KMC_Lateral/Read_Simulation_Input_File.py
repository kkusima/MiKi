class Read_Simulation_Input_File(Path)
    
    def _init_(self):
        data = open(Path, 'r').readlines()
        self.Temperature = float(data[5][24:31]) #Kelvin
        self.Total_Pressure = float(data[6][24:38]) 
        self.No_Gas_Species = int(data[8][23:])
        self.Gas_Species_Names = list((str(data[9][23:])).split())
        
        self.Gas_Species_Energies = [float(i) for i in data[10][23:].split()]
        self.Gas_Species_Mol_Weights = [float(i) for i in data[11][23:53].split()]
        self.Gas_Molar_Fractions = [float(i) for i in data[12][23:].split()]

        self.No_Surface_Species = int(data[14][23:])
        self.Surface_Species_Names = list((str(data[15][23:])).split())