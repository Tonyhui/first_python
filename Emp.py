class Emp:


    empCount = 0;

    def __init__(self, name , age ):
        self.name = name;
        self.age = age;
        Emp.empCount+=1;




    def getSalary(self):
        if( self.age > 40 ):
            return  9000;
        else:
            return  5000;





