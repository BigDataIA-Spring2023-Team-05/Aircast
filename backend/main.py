from fastapi import FastAPI
import uvicorn
from routers import user, service_plans
from config.db import Base, engine, SessionLocal
import repository.user as UserRepository
import repository.service_plans as servicePlans
from schemas.User import User
from models.User import Role
from pydantic import BaseModel, Field, EmailStr, validator

app =  FastAPI()
db = SessionLocal()


# create tables if not exist
def init_db():
    Base.metadata.create_all(bind=engine)
    servicePlans.create(1, 'Free', 10, db= db)
    servicePlans.create(2, 'Gold', 15, db= db)
    servicePlans.create(3, 'Platinum', 20, db= db)
    UserRepository.create(User(username='damg7245', email=EmailStr('rishab1300@gmail.com'), password='spring2023', planId=2, userType= Role.User), db= db)
    UserRepository.create(User(username='admin', email=EmailStr('mail@heyitsrj.com'), password='spring2023', planId=1, userType = Role.Admin), db= db)

    print("Initialized the db")

@app.on_event("startup")
async def startup():

    # register user router
    app.include_router(user.router)
    app.include_router(service_plans.router)


    init_db()
    

# define a default route
@app.get('/')
def index():
    return 'Success! APIs are working!'


if __name__ == '__main__':
    # start the server
    uvicorn.run(app, host='127.0.0.1', port=8000)