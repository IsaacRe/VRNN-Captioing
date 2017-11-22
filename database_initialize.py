from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime
from database_setup import *
 
engine = create_engine('sqlite:///course.db')
# Bind the engine to the metadata of the Base class so that the
# declaratives can be accessed through a DBSession instance
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
# A DBSession() instance establishes all conversations with the database
# and represents a "staging zone" for all the objects loaded into the
# database session object. Any change made against the objects in the
# session won't be persisted into the database until you call
# session.commit(). If you're not happy about the changes, you can
# revert all of them back to the last commit by calling
# session.rollback()
session = DBSession()

session.query(Input).delete()

input1 = Input(name="/static/step1/image_1.jpg",
	ground_true="<start> two men appear to play a game via the wii console . <end>")
session.add(input1)
session.commit()


input2 = Input(name="/static/step1/image_2.jpg",
	ground_true="<start> snowboarder cuts his way down a ski slope . <end>")
session.add(input2)
session.commit()


input3 = Input(name="/static/step1/image_3.jpg",
	ground_true="<start> a vegetable pizza on the edge of a table <end>")
session.add(input3)
session.commit()


print "Your database has been populated with some datas"

