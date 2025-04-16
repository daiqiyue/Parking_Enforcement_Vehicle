import mysql.connector
from mysql.connector import errorcode
def main():
    plate_num = 'QD7777'
    try:
        cnx = mysql.connector.connect(user = 'root', password = 'Dqy20030729@', host = 'localhost', database = "license_plate")
        print("Successfully connected to the database")
        cursor = cnx.cursor()
        query = ("select * from plate_info where plate_num = %s")
        cursor.execute(query, ("VL199",))
        result = cursor.fetchmany(size=1)
        print(result)   
        print(f"result type is {type(result)}")
        if not result:
            print("none")
        
        #for element in cursor:
        #    print(f"element is: {element}")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("The database doesn't exist")
    


    cnx.close()

if __name__ == "__main__":
    main()