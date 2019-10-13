# **Project Lullaby**

## Team: Erik, Matt, Joey, Mike

-------

What if parents and babies could get a better night's sleep? With Lullaby, we designed a web application that helps babies and parents get a better night's sleep by notifying them when their baby is crying and providing them with data to optimize their sleep strategies.

Inspired by Google's human voice libary https://developers.google.com/assistant/tools/sound-library/human-voices and their voice API.


## Primary clients:
* Parents/caregivers
* Babies - newborn & infants

## Main Tasks:

1. Detect Cries
    1. Capture audio
    2. Process captured audio
    3. Notify guardian
2. Handle Responses
    1. Determine gaurdian for response
    2. Address reasons for wake up

## **How it Works:**

### **Devices:**
* 1.A: An IoT or Home Device for baby's room
* 1.B: Caregiver's IoT / Home device / mobile
* Cloud

### **Implementation details**

IOT / Home Device

* Train on ML to recognize / distinguish infant crying sounds using Keras and Tensorflow
    * Record & track sounds coming from baby's room
    * Process audio
    * Generate label (string ? with label identifying sound)
    * Send label to appropriate devices
        * Back to device in baby room
            * change white noise (frequency / volume)
        * Alert to caregiver(s)
        * Record event in database with details ()
            * DETAILS: 
                * length of cry
                * type of sound (other baby generated sounds / movement learnable)
                * which caregiver alerted

* Caregiver device:
    * Alert - baby is crying
    * Trigger recording
        * length of time before they wake to respond
        * Time of wake
        * time to sooth baby / get back to sleep
        * number of times woken up 
            * that evening
            * aggregatable
    * NOTE: could be audio and video


## TODO
- [ ] device use cycle(s) - arrows that show pathway of "objects" & "actions"
- [ ] Notes / quotes on research / importance of sleep
- [ ] DataViz for Dashboard




 # Website
 https://lullabyzzz-20191013090820.azurewebsites.net/
 
 
 # Documentation
 
 https://hackmd.io/MPnwtfkkT0aCi30UjaEahQ?view
 
 
 
 ## Getting Started

In the root directory of the project...

1. Install node modules `yarn install` or `npm install`.
2. Start development server `yarn start` or `npm start`.


In Windows:

1. Install ffmpeg at 'https://www.ffmpeg.org/'
2. Change Path to C:/file/location/bin/ffmpeg.exe

## Next Steps


### Sample Data

Replace the sample data stored in /server/sampleData.js.
Replace the default images stored in /src/images.



### Adding a New Page

1. Create a folder in `/src/components` with your react components.
2. Add a route for your page to `/src/App.js`.
3. Add a button to the navigation bar in `/src/components/NavBar/index.js`.



### Cosmos Database

**Do Not share the keys stored in the .env file publicly.**
The Cosmos database will take approximately 5 minutes to deploy. Upon completion of deployment,
a notification will appear in VS Code and your connection string will be automatically added in
the .env file. The schema and operations for the Cosmos database are defined in `/server` folder.
Additional documentation can be found here: [Cosmos Docs](https://github.com/Microsoft/WebTemplateStudio/blob/dev/docs/services/azure-cosmos.md).

### Deployment

If you selected Azure App Service when creating your project, follow these steps:

1. Press `Ctrl + Shift + P` in Windows/Linux or `Shift ⇧ + Command ⌘ + P` in Mac and type/select `Web Template Studio: Deploy App` to start deploying your app.
2. After your project is built, click on "server" in the pop up on the top middle section of your screen, and then click "Deploy" on the window pop up.
3. Once the deployment is done, click "Browse website" in the notification window on the lower right corner to check out your newly deployed app.

If you did not select Azure App Service and want to create a new Azure App Service web app, follow these steps:

1. Press `Ctrl + Shift + P` in Windows/Linux or `Shift ⇧ + Command ⌘ + P` in Mac and type/select `Azure App Service: Create New Web App...` to create a new web app.
   - Select your subscription
   - Enter your web app name
   - Select Linux as your OS
   - Select Node.js 10.14 for a Node/Express application, Python 3.7 for a Flask application
2. Once the creation is done, click "Deploy" in the notification window on the lower right corner.
   - Click "Browse" on the top middle section of your screen and select the server folder within your project
   - Click "Yes" in the notification window on the lower right corner (build prompt)
   - Click "Deploy" on the window pop up
   - Click "Yes" in the notification window on the lower right corner again
3. Once the deployment is done, click "Browse website" in the notification window on the lower right corner to check out your newly deployed app.

Consider adding authentication and securing back-end API's by following [Azure App Service Security](https://docs.microsoft.com/en-us/azure/app-service/overview-security).

Full documentation for deployment to Azure App Service can be found here: [Deployment Docs](https://github.com/Microsoft/WebTemplateStudio/blob/dev/docs/deployment.md).

## File Structure

The front-end is based on [create-react-app](https://github.com/facebook/create-react-app).

The back-end is based on [Express Generator](https://expressjs.com/en/starter/generator.html).

The front-end is served on http://localhost:3000/ and the back-end on http://localhost:3001/.

```
.
├── audio-classification/ - Express server that provides API routes and serves front-end
│ ├── clean/ - Handles all interactions with the cosmos database
│ ├── models/
│ ├── oggfiles/ - Adds middleware to the express server
│ ├── pickles/
│ ├── wavfiles/ - Handles API calls for routes
│ ├── cries.csv - input file for training
│ ├── demo.csv - input file for demo
│ ├── prediction.csv - results of prediction.py
│ ├── cfg.py - configuration options for the program
│ ├── model.py - creates a trained ML model to predict sound
│ ├── prediction.py - classifies sound files using the trained model
│ └── requirements.txt - modules required for running the program
├── server/ - Express server that provides API routes and serves front-end
│ ├── mongo/ - Handles all interactions with the cosmos database
│ ├── routes/ - Handles API calls for routes
│ ├── app.js - Adds middleware to the express server
│ ├── sampleData.js - Contains all sample text data for generate pages
│ ├── constants.js - Defines the constants for the endpoints and port
│ └── server.js - Configures Port and HTTP Server
├── src - React front-end
│ ├── components - React components for each page
│ ├── App.jsx - React routing
│ └── index.jsx - React root component
├── .env - API Keys
└── README.md
```

## Additional Documentation


- React - https://reactjs.org/
- React Router - https://reacttraining.com/react-router/

- Bootstrap CSS - https://getbootstrap.com/
- Express - https://expressjs.com/


- Mongo/Mongoose - https://mongoosejs.com/docs/guide.html
- Cosmos DB - https://docs.microsoft.com/en-us/azure/cosmos-db/mongodb-mongoose

  This project was created using [Microsoft Web Template Studio](https://github.com/Microsoft/WebTemplateStudio).
