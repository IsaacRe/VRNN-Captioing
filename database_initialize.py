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

input1 = Input(id=1,path="/static/step1/image_1.jpg",
	ground_true="<start> Several planes lined up on a runway during a cloudy day. <end>",
	name = "airplane")
session.add(input1)
session.commit()


input2 = Input(id=2,path="/static/step1/image_2.jpg",
	ground_true="<start> A balding man is holding a black microphone. <end>",
	name = "bold man with microphone")
session.add(input2)
session.commit()


input3 = Input(id=3,path="/static/step1/image_3.jpg",
	ground_true="<start> A bald man in a suit looking down at something <end>",
	name = "bold man in black suit")
session.add(input3)
session.commit()


print "Your database has been populated with some datas"

