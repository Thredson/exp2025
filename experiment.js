// Initialize jsPsych
const jsPsych = initJsPsych({
    show_progress_bar: true,
    auto_update_progress_bar: true,
    on_finish: function(data) {
        console.log('Experiment fully completed');
    }
});

// TODO: FILL OUT APPROX. EXPERIMENT LENGTH

var time = '10-15 minutes';


const consent = {

type: jsPsychHtmlButtonResponse,

stimulus: '<h2><b>Consent Form</b></h2> <div style="text-align:left;' +

'background-color:lightblue; padding:20px; max-width:900px;">' +

'<p><b>Description:</b> Welcome! You are invited to participate' +

' in a research study in cognitive psychology. You will be asked' +

' to perform tasks on a computer which may include:' +

' looking at images, choose between options. ' +

'You may be asked a number of' +

// careful about single quotes

" different questions about making judgments and intepreting" +

" people's actions. All information collected will remain" +

' confidential. </p>' +

`<strong>Risks, Benefits and Data Confidentiality</strong>

<br>

There are no known risks in participating in this study, and no health or cognitive benefits. Whilst your data will be provided anonymously, at the point of data collection, your responses in the survey could, theoretically, be linked back to you via your Prolific ID, or your IP address. 

<br>

The former is collected to enable your payment, the latter to ensure that there are no duplicate responses in the database. After we have used the data for this purpose, this information will be deleted from the data file. The data will subsequently be stored anonymously, such that your individual responses will not be traceable back to you. 

All personal information will remain confidential and the data gathered will be stored anonymously and securely. It will not be possible to identify you in any publications. Any anonymised research data may be shared with, and used by, others for future research.

<br><br>

<strong>Participation or Withdrawal</strong>

<br>

Your participation in the study is voluntary. You may decline to answer any question and have the right to withdraw from participation at any time. Withdrawal will not affect your relationship with University College London in any way. 

Simply close your browser if you wish to withdraw.

<br><br>

<strong>**The study has been processed by the Research Ethics Committee at University College London and the study number is 0497.**</strong></div>` +


'<p> Do you agree with the terms of the experiment as explained' +

' above? </p>',

choices: ['I agree']

}
// let rewardCount = 0;

// Experiment parameters
const TRIAL_DURATION = null; // infinite duration until response
const TRAINING_BLOCKS = 8;
const TESTING_BLOCKS = 2;

// Randomly assign participant to feature/non-feature condition
const CONDITION = Math.random() < 0.5 ? 'feature' : 'non-feature';
// const CONDITION = 'feature';
console.log('Participant assigned to:', CONDITION, 'condition');

// Image stimuli (A through F)
function createRandomizedImages() {
    // Original image files
    const imageFiles = ['imageA.png', 'imageB.png', 'imageC.png', 'imageD.png', 'imageE.png', 'imageF.png'];
    
    // Shuffle the image files
    const shuffledFiles = jsPsych.randomization.shuffle(imageFiles);
    
    // Letters that will be used in the experiment
    const letters = ['A', 'B', 'C', 'D', 'E', 'F'];
    
    // Create the randomized mapping
    const randomizedImages = {};

    letters.forEach((letter, index) => {
        randomizedImages[letter] = `${CONDITION}/${shuffledFiles[index]}`;
    })
    
    return randomizedImages;
}

// Create the randomized image assignment
const IMAGES = createRandomizedImages();

// Image randomization record
const randomization_record = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: '',
    trial_duration: 1,
    choices: "NO_KEYS",
    data: {
        trial_type: 'randomization_record',
        condition: CONDITION,
        randomized_images: IMAGES
    }
};

// Create timeline
let timeline = [];

// Welcome screen
const welcome = {
    type: jsPsychInstructions,
    pages: [
        `<h1>Welcome to the Experiment</h1>
         <p>press Next to begin</p>`
    ],
    show_clickable_nav: true
};

// Main instructions
const instructions = {
    type: jsPsychInstructions,
    pages: [
        `<h1>Welcome to the Experiment</h1>
         <p>Thank you for participating in this study! 
         <br> In this study, you will respond by <strong>selecting one of two pictures</strong> in each round. 
         <br> Each round, based on your response, you will receive either <strong> no reward</strong>
         or <strong>a 0.3-cent bonus</strong>. Therefore, try your best to choose responses that 
         may lead to <strong>higher rewards!</strong>
         <br>
         <br> There will be a training phase and a testing phase. 
         <br>In the training phase, 
         you will <strong>receive feedback</strong> indicating whether you received a reward. 
         <br>In the testing phase, 
         there will be <strong>no feedback</strong>, but the underlying rule remains the same, and your bonus will 
         continue to accumulate.</p>
         <p>You will see pairs of images and choose one using:</p>
         <ul>
            <li><strong>F key</strong> = Choose LEFT image</li>
            <li><strong>J key</strong> = Choose RIGHT image</li>
         </ul>
         <p>Ready to begin?</p>`
    ],
    show_clickable_nav: true
};

// Track when each stimulus was last chosen (for reward calculation)
let UnchosenTrial = {
    A: 0, B: 0, C: 0, D: 0, E: 0, F: 0
};
let trialCounter = 0;

// Function to create reward pattern: P(reward) = 1-(1-0.3)^(n+1)
function getReward(chosenImage, unchosenImage) {

    if (!chosenImage) return false; // No choice made

    UnchosenTrial[unchosenImage] += 1;
    
    // Calculate n
    const n = UnchosenTrial[chosenImage];
    
    // reset last chosen trial
    UnchosenTrial[chosenImage] = 0;
    
    // Apply formula
    const rewardProbability = 1 - Math.pow(1 - 0.3, n + 1);
    
    // Generate random number and compare with probability
    const random = Math.random();
    const reward = random < rewardProbability;

    console.log(`Trial ${trialCounter + 1}: Chose ${chosenImage}, Didn't choose ${unchosenImage} n=${n}, P(reward)=${rewardProbability.toFixed(3)}, got reward: ${reward}`);

    return reward;
}

// Create training pairs: AB, BC, CD, DE, EF + reversed
function createTrainingPairs() {
    const basePairs = [
        ['A', 'B'], ['B', 'C'], ['C', 'D'], 
        ['D', 'E'], ['E', 'F']
    ];
    
    const allPairs = [];
    basePairs.forEach(pair => {
        allPairs.push(pair); // Original order
        allPairs.push([pair[1], pair[0]]); // Reversed order
    });
    
    return jsPsych.randomization.shuffle(allPairs);
}

// Create testing pairs: all combinations A-F + reversed  
function createTestingPairs() {
    const letters = ['A', 'B', 'C', 'D', 'E', 'F'];
    const allPairs = [];
    
    for (let i = 0; i < letters.length - 1; i++) {
        for (let j = i + 1; j < letters.length; j++) {
            allPairs.push([letters[i], letters[j]]); // Original
            allPairs.push([letters[j], letters[i]]); // Reversed
        }
    }
    
    return jsPsych.randomization.shuffle(allPairs);
}

// Create choice trial
function createChoiceTrial(pair, blockNum, phase) {
    const leftImage = pair[0];
    const rightImage = pair[1];
    
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function() {
            return `
                <div class="phase-indicator">${phase.toUpperCase()} PHASE</div>
                <div class="progress-info">Choose an image: F = Left, J = Right</div>
                <div class="image-pair-container">
                    <div class="image-choice">
                        <img src="${IMAGES[leftImage]}" alt="Image ${leftImage}">
                        <div class="key-label">F</div>
                    </div>
                    <div class="image-choice">
                        <img src="${IMAGES[rightImage]}" alt="Image ${rightImage}">
                        <div class="key-label">J</div>
                    </div>
                </div>
            `;
        },
        choices: ['f', 'j'],
        trial_duration: TRIAL_DURATION,
        data: {
            phase: phase,
            block: blockNum,
            left_image: leftImage,
            actual_left_image: IMAGES[leftImage].replace('non-feature/', ''),
            right_image: rightImage,
            actual_right_image: IMAGES[rightImage].replace('non-feature/', ''),
            pair: `${leftImage}-${rightImage}`,
            condition: CONDITION
        },
        /*
        on_load: function() {
            // Countdown timer
            let timeLeft = 4;
            const timerElement = document.getElementById('timer');
            
            const countdown = setInterval(function() {
                timeLeft--;
                if (timerElement) {
                    timerElement.textContent = timeLeft;
                }
                if (timeLeft <= 0) {
                    clearInterval(countdown);
                }
            }, 1000);
        },
        */
        on_finish: function(data) {
            // Record choice
            if (data.response === 'f') {
                data.chosen_image = data.left_image;
                data.unchosen_image = data.right_image;
                data.chosen_side = 'left';
            } else if (data.response === 'j') {
                data.chosen_image = data.right_image;
                data.unchosen_image = data.left_image;
                data.chosen_side = 'right';
            } else {
                data.chosen_image = null;
                data.unchosen_image = null;
                data.chosen_side = null;
            }
            
            // Determine reward
            data.reward = getReward(data.chosen_image, data.unchosen_image);
            // if (data.reward == true) {
            //     rewardCount += 1; // Increment reward count
            // }
            data.reaction_time = data.rt;
            data.trial_number = trialCounter;
            
            // Increment trial counter
            trialCounter++;
        }
    };
}

// Create feedback trial for training phase
function createFeedbackTrial(phase) {
    if (phase == 'training') {
        return {
            type: jsPsychHtmlKeyboardResponse,
            stimulus: function() {
                const lastTrial = jsPsych.data.getLastTrialData().values()[0];
                
                if (lastTrial.reward) {
                    return `<div class="feedback reward">REWARD! +0.3 cents</div>`;
                } else {
                    return `<div class="feedback no-reward">No reward</div>`;
                }
            },
            choices: "NO_KEYS",
            trial_duration: 1500
        };
    }   
    else if (phase == 'testing') {
        // No feedback in testing phase
        return {
            type: jsPsychHtmlKeyboardResponse,
            stimulus: `<div class="feedback no-feedback">response recorded</div>`,
            choices: "NO_KEYS",
            trial_duration: 1500
        };
    }   
    else {
        // Default case, no feedback
        throw new Error("Invalid phase for feedback trial: " + phase);
    }
}

// Block start message
timeline.push({
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `<h3>Training Block</h3>
                   <p>Press any key to start this block.</p>`,
        choices: "ALL_KEYS"
    });

// Build training phase
for (let block = 1; block <= TRAINING_BLOCKS; block++) {
    // Create trials for this block
    const pairs = createTrainingPairs();
    pairs.forEach(pair => {
        timeline.push(createChoiceTrial(pair, block, 'training'));
        timeline.push(createFeedbackTrial('training'));
    });
    
    // breaks between blocks  
    if (block == 4) {
        timeline.push({
            type: jsPsychHtmlKeyboardResponse,
            stimulus: `<p>Take a break for a few seconds if needed.</p>
                       <p>Press any key to continue</p>`,
            choices: "ALL_KEYS"
        });
    }
    
}

// Transition to testing phase
timeline.push({
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `<h2>Training Complete!</h2>
               <p>Now this is the testing phase, there will be no feedback, 
               but the underlying rule remains the same, and your bonus 
               will continue to accumulate.</p>
               <p>Press any key to begin the testing phase.</p>`,
    choices: "ALL_KEYS"
});

// Build testing phase  
for (let block = 1; block <= TESTING_BLOCKS; block++) {
    timeline.push({
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `<h3>Testing Block</h3>
                   <p>Press any key to start this block.</p>`,
        choices: "ALL_KEYS"
    });
    
    const pairs = createTestingPairs();
    pairs.forEach(pair => {
        timeline.push(createChoiceTrial(pair, block, 'testing'));
        timeline.push(createFeedbackTrial('testing'));
    });
}

// Post-experiment questionnaire
const questionnaire = {
    type: jsPsychSurveyText,
    preamble: `<h2>Post-Experiment Questionnaire</h2>
               <p>Thank you for completing the experiment!
                Please answer the following questions about your experience.</p>`,
    questions: [
        {
            prompt: "*What do you think is the rule for getting the reward?",
            name: 'rules',
            rows: 4,
            required: true
        },
        {
            prompt: "*What strategy did you use to make your choices?",
            name: 'strategy', 
            rows: 4,
            required: true
        },

    ]
};

const strategychoice = {
    type: jsPsychSurveyMultiChoice,
    preamble: `<h2>Post-Experiment Questionnaire</h2>`,
    questions: [
         {
            prompt: "*Which strategy best describes how to get more reward in this experiment?",
            name: 'strategy_choice',
            options: function() {
                const strategies = [
                    "There is a ranking between six pictures. For example, Picture 1 > Picture 2 > Picture 3... A picture with a relative higher rank in the present pair should be preferred.",
                    "Pictures with some specific features should be preferred",
                    "The picture you picked previously should not be picked on the next occasion, which means the picture not picked previously should now be picked",
                    "Completely random"
                ];
                const shuffledStrategies = jsPsych.randomization.shuffle(strategies);                
                return shuffledStrategies;
            },
            required: true
        },
        {
            prompt: `*Did you take notes or use any external tools that helped you during the experiment?
             \n This option will not affect your reward.`,
            name: 'took-notes',
            options: [
                'yes', 'no'   
            ],
            required: true
        },
    ],  
};

const demographics = {
    type: jsPsychSurveyText,
    preamble: `<h2>Demographics</h2>`,
    questions: [
        {
            prompt: "*please enter your age.",
            name: 'age',
            required: true
        },
        {
            prompt: "*please enter your gender.",
            name: 'gender',
            required: true
        },
        {
            prompt: "*please enter your highest level of education.",
            name: 'education',
            required: true
        },
        {
            prompt: "Any additional comments or feedback?",
            name: 'comments',
            rows: 4,
            required: false
        }
    ]
};

// Final screen
const final_screen = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `<h2>Experiment Complete!</h2>
               <p>Thank you for your participation in this study.</p>
               <br>
                <div id="upload-status" style="margin: 20px 0; padding: 15px; border-radius: 5px; background-color: #f8f9fa;">
                    <p>📤 Uploading data to server...</p>
                </div>
                <div id="download-section" style="display: none;">
                    <p><strong>Server upload failed. Please download your data manually:</strong></p>
                    <button id="download-btn" style="
                        background-color: #dc3545;
                        color: white;
                        border: none;
                        padding: 15px 30px;
                        font-size: 18px;
                        border-radius: 5px;
                        cursor: pointer;
                        margin: 20px 0;
                    " onmouseover="this.style.backgroundColor='#c82333'" 
                       onmouseout="this.style.backgroundColor='#dc3545'">
                        📥 Download Data File
                    </button>
                </div>
                <p id="exit-instruction"><em>Data is being saved automatically.</em></p>`;
    },
    choices: "ALL_KEYS",
    on_start: function() {
        // Prepare data
        const allData = jsPsych.data.get().values();
        const jsonData = JSON.stringify(allData, null, 2); // Same format for both DataPipe and local
        
        // Create filename with timestamp and condition
        const timestamp = new Date().toISOString().slice(0,19).replace(/:/g, '-');
        const filename = `data_${CONDITION}_${timestamp}.json`;
        
        // Send JSON data to DataPipe
        fetch("https://pipe.jspsych.org/api/data/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "*/*",
            },
            body: JSON.stringify({
                experimentID: "pgVYBO7khUHm",
                filename: filename,
                data: jsonData,
            }),
        })
        .then(response => response.json())
        .then(result => {
            console.log('Data successfully uploaded to DataPipe:', result);
            
            // Update upload status to success
            const statusDiv = document.getElementById('upload-status');
            if (statusDiv) {
                statusDiv.innerHTML = '<p style="color: green;">✅ Data successfully uploaded to server!</p>';
                statusDiv.style.backgroundColor = '#d4edda';
            }
            
            // Update exit instruction
            const exitInstruction = document.getElementById('exit-instruction');
            if (exitInstruction) {
                exitInstruction.innerHTML = '<em>Data saved successfully! You may close this tab to exit.</em>';
            }
        })
        .catch(error => {
            console.error('Error uploading to DataPipe:', error);
            
            // Update upload status to failed
            const statusDiv = document.getElementById('upload-status');
            if (statusDiv) {
                statusDiv.innerHTML = '<p style="color: red;">❌ Server upload failed.</p>';
                statusDiv.style.backgroundColor = '#f8d7da';
            }
            
            // Show download section
            const downloadSection = document.getElementById('download-section');
            if (downloadSection) {
                downloadSection.style.display = 'block';
            }
            
            // Update exit instruction
            const exitInstruction = document.getElementById('exit-instruction');
            if (exitInstruction) {
                exitInstruction.innerHTML = '<em>Please download your data before closing this tab.</em>';
            }
            
            // Set up download button
            setTimeout(function() {
                const downloadBtn = document.getElementById('download-btn');
                if (downloadBtn) {
                    downloadBtn.addEventListener('click', function() {
                        // Create and download JSON file
                        const blob = new Blob([jsonData], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = filename;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                        
                        // Update button to show download completed
                        downloadBtn.innerHTML = '✅ Data Downloaded Successfully!';
                        downloadBtn.style.backgroundColor = '#28a745';
                        downloadBtn.disabled = true;
                        downloadBtn.style.cursor = 'default';
                        
                        // Update exit instruction
                        const exitInstruction = document.getElementById('exit-instruction');
                        if (exitInstruction) {
                            exitInstruction.innerHTML = '<em>Data saved! You may now close this tab or press any key to exit.</em>';
                        }
                        
                        console.log('Data downloaded locally:', filename);
                    });
                }
            }, 100);
        });
    },
    on_finish: function() {
        window.close(); // Close the tab after completion
    }
};

// Add all components to timeline
// timeline.unshift(strategychoice);
timeline.unshift(final_screen);
timeline.unshift(randomization_record);
timeline.unshift(instructions);
timeline.unshift(consent);
timeline.push(questionnaire);
timeline.push(strategychoice);
timeline.push(demographics);
timeline.push(final_screen);

// Run the experiment

jsPsych.run(timeline);






