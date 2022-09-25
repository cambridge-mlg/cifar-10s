var timeline = []

/* init connection with pavlovia.org */
var pavlovia_init = {
  type: "pavlovia",
  command: "init"
};
timeline.push(pavlovia_init);

// capture info from Prolific
// help from: https://www.jspsych.org/overview/prolific/
var prolific_id = jsPsych.data.getURLVariable('PROLIFIC_PID');
var study_id = jsPsych.data.getURLVariable('STUDY_ID');
var session_id = jsPsych.data.getURLVariable('SESSION_ID');
// subj id help from: https://www.jspsych.org/7.0/overview/data/index.html
// generate a random subject ID with 15 characters
var subject_id = jsPsych.randomization.randomID(15);
jsPsych.data.addProperties({
    subject: subject_id,
    prolific_id: prolific_id,
    study_id: study_id,
    session_id: session_id
});

var official_run = true

// pick a random condition for the subject at the start of the experiment
// help from: https://www.jspsych.org/overview/prolific/
// based on our total number of batches <--- note: can subset if we need to run some a few more
var num_batches = 40

function numberRange (start, end) {
  // from: https://stackoverflow.com/questions/3895478/does-javascript-have-a-method-like-range-to-generate-a-range-within-the-supp
  return new Array(end - start).fill().map((d, i) => i + start);
}

var conditions = numberRange(0, num_batches);

var condition_num = jsPsych.randomization.sampleWithoutReplacement(conditions, 1)[0];

console.log(condition_num)

// record the condition assignment
jsPsych.data.addProperties({
  condition: condition_num
});

// index into batched mixup images
// javascript loading help from: https://github.com/jspsych/jsPsych/discussions/705
var imgs = batches[0][condition_num]

// cifar10 classes
var classNames = ['Airplane', 'Automobile', 'Bird','Cat', 'Deer','Dog', 'Frog', 'Horse', 'Ship', 'Truck']
// shuffle per person, but same order w/in annotator
classNames=jsPsych.randomization.shuffle(classNames)
// classNames.push("Other") // add an "other"/extra class option at the end

var remClasses = []
var noneOption = "No Alternative"

var headerInstructionTxt = '<center>Imagine 100 crowdsourced workers are asked to <strong>identify what category the image below belongs to</strong>.</center><br></br>'

console.log("images: ", imgs.length, " batch: ", condition_num)

// consent form help from: https://gitlab.pavlovia.org/beckerla/language-analysis/blob/master/html/language-analysis.js
// sample function that might be used to check if a subject has given consent to participate.
var check_consent = function(elem) {
    if ($('#consent_checkbox').is(':checked') && $('#read_checkbox').is(':checked') && $('#age_checkbox').is(':checked')) {
	return true;
    }
    else {
	alert("If you wish to participate, you must check the boxes in the Consent Form.");
	return false;
    }
    return false;
};
var consent = {
    type:'external-html',
    url: "consent.html",
    cont_btn: "start",
    check_fn: check_consent
}

if (official_run){
timeline.push(consent)
}


var num_rerun = 2 // e.g., to check consistency

var num_show = imgs.length + num_rerun

var num_pages_per_img = 1 // to handle num pages per img

var progress_bar_increase = 1 / (num_show * num_pages_per_img)


var instructions = {
	type: "instructions",
	pages: ['<p> Welcome! </p> <p> We are conducting an experiment about how people express uncertainty over images. Your answers will be used to inform machine learning and human-computer interaction work. </p>' +
	'<p> This experiment should take at most <strong>25 minutes</strong>. </br></br> You will be compensated at a base rate of $8/hour for a total of <strong>$3.34</strong>, which you will receive as long as you complete the study.</p>',
			'<p> We take your compensation and time seriously! The email for the main experimenter is <strong>cambridge.mlg.studies@gmail.com</strong>. </br></br> Please write this down now, and email us with your Prolific ID and the subject line <i>Human experiment compensation</i> if you have problems submitting this task, or if it takes much more time than expected. </p>',
			'<p> In this experiment, you will be seeing <i>images</i> of objects.</p>'+
			'<p> Each image depicts an instance of some <i>category</i>, e.g., a dog, a truck, an airplane.</p>' +
      '<p> You will have <strong>multiple tasks per image</strong>, including coming up with the most probable, second most probable, and improbable categories for the image. Along with associated estimates of the probabilities of those categories being the correct category of the image. </p>',
			'<p> Your first task then will be to <strong>select the most probable category</strong> represented in the image.</p>' +
			'<p> We ask that you imagine that <strong>100 crowdsourced workers</strong> are doing this task. Consider how they may respond.</p>' +
			'<p> You will <strong>select</strong> the category that you think these 100 crowdsourced workers would think is <strong>most likely to be the true class in the image</strong> by clicking a radio button.</p>' + //If the category you see in the image is not included in the list of categories (Cat, Dog, Ship, etc.), please selected Other.</p>' +
			'<p> Please then <strong>type</strong> in the text box the <strong>percent probability you think the other annotators would assign</strong> to that category being the true category.</p>',
			'<p> You will also be asked to follow a similar <strong>click-and-type response</strong> for what you think the crowdsourced workers would think the <strong>alternatively most probable</strong> true category of the image is.</p>' +
			'<p> If you think that the category that you selected as most probable in the first question is the only likely category (e.g., that category is 100\% sure to be the correct category), click the "' + noneOption + '" option. </p>' +
			'<p> However, if you do think the image could be showing a different category than what you selected previously (e.g., the image is most likely a dog, but could be a cat), please enter the associated probability you think for that class.</p>' +
			'<p> You <strong>do not need to worry</strong> that the percent probabilities you type sum to 100. We will normalize after.</p>',
			'<p> Finally, you will best asked to click <i>all</i> categories you think are <strong>definitely not</strong> in the image.' +
			'<p> Click <i>all</i> categories that you think would be assigned <i>zero probability</i> as being the true category (e.g., there is a <strong>0\% chance that the categories selected are what are shown in the image</strong>).</p>' +
      '<p> For instance, you may think that an image would be categorized as an automobile, or maybe a truck but definitely not as a dog, cat, or frog. In that case, please click dog, cat, <i>and</i> frog.</p>', //<p>Please skip this question only if you think that the 100 crowdsourced workers would think that all categories could possibly be the true category shown in the image.</p>',

			'<p> The category of some images may be obvious, and you may have high confidence as to what the true class is. </p>' +
			'<p> For others, it may be difficult to determine. Please try your best. </p>',

			'<p> You will receive a <strong>bonus</strong> of up to a rate of $9/hour (+$0.41) if your responses most closely match what other annotators provide.</p>' +
			'<p> We therefore encourage you to select categories and specify probabilities that you think others (e.g., 100 crowdsourced workers) would assign to the image, and provided you demonstrated sufficient effort, you will attain the bonus. </p>',

			'<p> You will see a total of <strong>' + num_show + ' images</strong>.</p>' +
			'<p> When you are ready, please click <strong>\"Next\"</strong> to complete a quick comprehension check, before moving on to the experiment. </p>'],
	show_clickable_nav: true
};

var correct_task_description = "The most likely category, or categories, of an image and the associated probability of the selected categories."

var correct_perspective_description = "100 crowdsourced workers."
var incorrect_perspective_description = "Your own."
var incorrect_perspective_description2 = "Your dog's."

var comprehension_check = {
    type: "survey-multi-choice",
    preamble: ["<p align='center'>Check your knowledge before you begin. If you don't know the answers, don't worry; we will show you the instructions again.</p>"],
    questions: [
        {
            prompt: "What will you be asked to determine in this task?",
            options: [correct_task_description, "The similarity between images of cats and trucks.", "The funniness of jokes and your confidence in your estimate.",],
            required: true
        },

        {
            prompt: "Whose perspective are you considering when making your response?</i>",
            options: [incorrect_perspective_description, incorrect_perspective_description2, correct_perspective_description],
            required: true
        },
    ],
    on_finish: function (data) {
        var responses = data.response;

        console.log(data)

        console.log(data.response)

        if (responses['Q0'] == correct_task_description && responses['Q1'] == correct_perspective_description) {
            familiarization_check_correct = true;
        } else {
            familiarization_check_correct = false;
        }
    }
}

var familiarization_timeline = [instructions, comprehension_check]

var familiarization_loop = {
    timeline: familiarization_timeline,
    loop_function: function (data) {
        return !familiarization_check_correct;
    }
}

if (official_run){
timeline.push(familiarization_loop)
}

var final_instructions = {
    type: "instructions",
    pages: ['<p> Now you are ready to begin! </p>' +
    '<p> Please click <strong>\"Next\"</strong> to start the experiment. Note, it may take a moment for each image to load. </p>' +
    '<p> Thank you for participating in our study! </p>'],
    show_clickable_nav: true
};
timeline.push(final_instructions)

// preload stimuli
var pre_load_imgs = []
for (var i = 0; i < imgs.length; i++){
    pre_load_imgs.push("imgs/" + imgs[i]['filename'])
}

// preload help from: https://www.jspsych.org/6.3/plugins/jspsych-preload/
var preload = {
    type: 'preload',
    images: pre_load_imgs,
    show_detailed_errors: true
}
if (official_run){
timeline.push(preload)
}

var imageWidth = 160
var imageHeight = 160

// help for various input forms from: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input
// and: https://www.jspsych.org/7.0/plugins/survey-html-form/

var ordered_idxs = [] // save order used

var main_page = {

    type: 'survey-html-form',
    preamble: function () {
                var img = jsPsych.timelineVariable('filename')
                var label1 = jsPsych.timelineVariable("label1")
                var label2 = jsPsych.timelineVariable("label2")

                img = "imgs/" + img

                var custom_mixing_scale = [
                  "100\% "+ label1,
                  "50/50 " + label1 + " and " + label2,
                  "100\% " + label2
                ]

                return '<p> ' +
                headerInstructionTxt +
                '<p><img src=' + img + ' style="max-width:750px; max-height:750px;"></p>'
            },

    html: function() {
      var pretxt = "<p><center>What category do you think they would select as <strong>most probably</strong> being the true category of the image?</center></p>"

      // help for various input forms from: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input
      // and: https://www.jspsych.org/7.0/plugins/survey-html-form/
			// alignment help from: http://www.java2s.com/example/html-css/css-form/align-a-radio-buttons-text-under-the-button-itself.html

			var classForm = '<form action="">'
			for (var classIdx = 0; classIdx < classNames.length; classIdx++){
				var className = classNames[classIdx]
				classForm += '<div class="aligned-box"><input type="radio" id=mostProb'+className+' name=classSelect value=mostProb'+className+' required>'
				+ '<label for='+className+'>' + className + '</label></div>'
			}


      // reset other classes for next page
      remClasses = classNames

			classForm += "</p>"

      var probEnter = '<p><center>What <strong>percent probability (between 0 and 100)</strong>  do you think they would assign to the category you selected being the true category of the image?'

			probEnter += ' <input name="prob" type="text" required />%</center><p>'

      mostProbTxt = pretxt + classForm + probEnter


      var pretxt = "<p><center>What <strong>alternate</strong> category, if any, do you think they would select as being the <strong>second most probable</strong> of being the true category of the image? </center></p>" //Skip this question if you think there are no alternatives.</center></p>"

      // help for various input forms from: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input
      // and: https://www.jspsych.org/7.0/plugins/survey-html-form/
			var classForm = '<form action="">'
			for (var classIdx = 0; classIdx < classNames.length; classIdx++){
				var className = classNames[classIdx]
				classForm += '<div class="aligned-box"><input type="radio" id=secondProb'+className+' name=classSelect2 value=secondProb'+className+' required>'
				+ '<label for='+className+'>' + className + '</label></div>'
			}

      // add a none option
      classForm += '<div class="aligned-box"><input type="radio" id=secondProb'+noneOption+' name=classSelect2 value=secondProb'+noneOption+' required>'
      + '<label for='+noneOption+'>' + noneOption + '</label></div>'

			classForm += "</p>"

			var probEnter = '<p><center>If you selected an alternate category for the image, what <strong>percent probability (between 0 and 100)</strong> do you think they would assign to the category you selected being the true category represented in the image?' // </center></p>'

			probEnter += ' <input name="prob2" type="text" />%</center><p>'

      secondProbTxt = pretxt + classForm + probEnter

      var pretxt = "<p><center>Are there one or more categories you think the crowdsourced annotators would say are <strong>definitely not</strong> the true category of the image?</center></p>" +
      '<p><center>Please click <strong>ALL</strong> categories you think the annotators would say have <i>zero probability</i> of being the true category.</center></p>'


      // help for various input forms from: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input
      // and: https://www.jspsych.org/7.0/plugins/survey-html-form/

      var classForm = '<form action="">'
      for (var classIdx = 0; classIdx < classNames.length; classIdx++){
        var className = classNames[classIdx]
        classForm += '<div class="aligned-box"><input type="checkbox" id=improbClassSelect'+className+' name=improbClassSelect' + className + ' value=1>'
        + '<label for='+className+'>' + className + '</label></div>'
      }

      classForm += "</p>"

      improbTxt = pretxt + classForm

      var finalTxt = mostProbTxt + secondProbTxt + improbTxt
        return finalTxt
    }

}

var rating_task = {
    timeline: [main_page],
    timeline_variables: imgs,
    data: {
        filename: jsPsych.timelineVariable('filename'),
        task: 'spec_conf',
        subj_id: jsPsych.timelineVariable('id'),
				label: jsPsych.timelineVariable('label'),
        img_id: jsPsych.timelineVariable('example_idx'),
        data_split: jsPsych.timelineVariable('data_split')
    },
    sample: {
        type: 'custom',
        fn: function (t) {
            // t = set of indices from 0 to n-1, where n = # of trials in stimuli variable
            // returns a set of indices for trials

            ordered_idxs = jsPsych.randomization.shuffle(t)
            return ordered_idxs
        }
    },
    on_finish: function () {
        var curr_progress_bar_value = jsPsych.getProgressBarCompleted();
        jsPsych.setProgressBar(curr_progress_bar_value + progress_bar_increase);

        console.log(curr_progress_bar_value)
    }
}

timeline.push(rating_task);

// for consistency!
var rerun_rating_task = {
    timeline: [main_page],
    timeline_variables: imgs,
    data: {
        filename: jsPsych.timelineVariable('filename'),
        task: 'rerun_spec_conf',
        subj_id: jsPsych.timelineVariable('id'),
				label: jsPsych.timelineVariable('label'),
        img_id: jsPsych.timelineVariable('example_idx'),
        data_split: jsPsych.timelineVariable('data_split')
    },
		sample: {
        type: 'custom',
        fn: function (t) {
            // returns a set of indices for trials

            // just have this for a couple of few imgs (checks), help from: https://stackoverflow.com/questions/34883068/how-to-get-first-n-number-of-elements-from-an-array
            return [ordered_idxs[5], ordered_idxs[10]] // always show the 15th and 20th
        }
    },
    on_finish: function () {
        var curr_progress_bar_value = jsPsych.getProgressBarCompleted();
        jsPsych.setProgressBar(curr_progress_bar_value + progress_bar_increase);

        console.log(curr_progress_bar_value)
    }
}
timeline.push(rerun_rating_task);

var comments_block = {
    type: "survey-text",
    preamble: "<p>Thank you for participating in our study!</p>" +
    "<p>Click <strong>\"Finish\"</strong> to complete the experiment and receive compensation. If you have any comments about the experiment, please let us know in the form below.</p>",
    questions: [
        {prompt: "Were the instructions clear? (On a scale of 1-10, with 10 being very clear)"},
        {prompt: "How challenging was it to come up an associated probability of the categories you selected per image? (On a scale of 1-10, with 10 being very challenging)"},
				{prompt: "Did you use a particular strategy when determining which class was most probable?"},
				{prompt: "Were there instances where you would have liked to go back and change your answer after considering the second most probable class?"},
        {prompt: "Were there any particular qualities of images you considered when coming up with your response?", rows:5,columns:50},
        {prompt: "Do you have any additional comments to share with us?", rows: 5, columns: 50}],
    button_label: "Finish",
};
timeline.push(comments_block)

/* finish connection with pavlovia.org */
var pavlovia_finish = {
  type: "pavlovia",
  command: "finish",
};
timeline.push(pavlovia_finish);

// todo: update w/ proper prolific link!!
jsPsych.init({
    timeline: timeline,
    on_finish: function () {
        // send back to main prolific link
        // specify window.location with your own prolific link
        window.location = None
    },
    show_progress_bar: true,
    auto_update_progress_bar: false
});
