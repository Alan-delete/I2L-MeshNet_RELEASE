const DEFAULT_TIMESTAMP = 1 //
const d = new Date()
let time = null
let stopUpload = true
let action_record = []
let url = `${ngrok_url}realTimeUpload`
import { update_skeleton} from './draw.js'

let start_camera = document.querySelector("#start-camera")
let video = document.querySelector("#video")
let start_capture = document.querySelector('#start-capture')
let captured_image = document.querySelector('#captured-image')

let actionCounter = 0
let accuracyRecord = []
let one_click_joint_record = []
let startingTime
let imgList = []
let timestampList = []

//const and var related to continuous image appending
const IMAGE_BATCH = 8
const EST_DELAY = 1.5
const IMAGE_NUM_OVERHEAD = 4
const IMAGE_INTERVAL = EST_DELAY/IMAGE_BATCH/IMAGE_NUM_OVERHEAD
let newImgTimer = null

//start camera display
start_camera.addEventListener('click', async function(){
    if (navigator.mediaDevices.getUserMedia || navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia) {
        getUserMedia({video: { video: true, audio: false }}, success_func, error_func);
    } else {
        alert('media access not supported');
    }
})

// assume every loop takes EXACT 1000/30 miliseconds to complete 
function continue_show_scene() {
  let frame_index = 0;
  let start_idx = 0;
  let timeId = setInterval(() => {
    if (start_idx < (one_click_joint_record.length-1)) {
      // exceed time slot 
      let cur_time_diff = frame_index * 1000 / 30;
      
      if (cur_time_diff >=
        (one_click_joint_record[one_click_joint_record.length - 1].timestamp
          - one_click_joint_record[start_idx].timestamp)) {

        start_idx = one_click_joint_record.length - 1;
        frame_index = 0;
      }
      else {
            for (let i = start_idx + 1; i< one_click_joint_record.length; i++){

              let time_diff = one_click_joint_record[i].timestamp
                - one_click_joint_record[start_idx].timestamp;

              if (cur_time_diff <= time_diff) {
	              let coe = (time_diff - cur_time_diff)/(one_click_joint_record[i].timestamp - one_click_joint_record[i-1].timestamp);
		            // linear interpolation
	              let new_skeleton = one_click_joint_record[i]['smpl_joint_coords'].map( (inner, row_idx)=>inner.map( (ele, col_idx)=> coe * ele + (1-coe) * one_click_joint_record[i-1]['smpl_joint_coords'][row_idx][col_idx] ) )
	              // update new_skeleton 
                update_skeleton(new_skeleton, I2L_skeleton);
                frame_index++;
	              break;
	            }
            }       
      }

    }
    else {
      frame_index = 0;
    }

    
  },1000/30)
  
}


// currently 30 fps
function replay(record_data){
    let start = Date.now();
    let timerId = setInterval(()=>{
        let cur_time_diff = (Date.now() - start);
        if (cur_time_diff >= (record_data[record_data.length-1].timestamp - record_data[0].timestamp)){
            clearInterval(timerId);
        }
        else{
            for (let i = 1; i< record_data.length; i++){
	        let time_diff = record_data[i].timestamp - record_data[0].timestamp
	        if (time_diff >= cur_time_diff){
	            let coe = (time_diff - cur_time_diff)/(record_data[i].timestamp - record_data[i-1].timestamp);
		    // linear interpolation
	            let new_skeleton = record_data[i]['smpl_joint_coords'].map( (inner, row_idx)=>inner.map( (ele, col_idx)=> coe * ele + (1-coe) * record_data[i-1]['smpl_joint_coords'][row_idx][col_idx] ) )
	            // update new_skeleton 
		    update_skeleton(new_skeleton, I2L_skeleton)
	            break;
	        }
            }
        }
    }, 1000/30)
}

//for above function
function getUserMedia(constraints, success, error){
    if (navigator.mediaDevices.getUserMedia) {
        //最新的标准API
        navigator.mediaDevices.getUserMedia(constraints).then(success).catch(error);
    } else if (navigator.webkitGetUserMedia) {
        //webkit核心浏览器
        navigator.webkitGetUserMedia(constraints, success, error)
    } else if (navigator.mozGetUserMedia) {
        //firfox浏览器
        navigator.mozGetUserMedia(constraints, success, error);
    } else if (navigator.getUserMedia) {
        //旧版API
        navigator.getUserMedia(constraints, success, error);
    }
}

//for above function
function success_func(stream) {
    //兼容webkit核心浏览器
    var CompatibleURL = window.URL || window.webkitURL;
    //将视频流设置为video元素的源
    // console.log(stream);
    //video.src = CompatibleURL.createObjectURL(stream);
    video.srcObject = stream;
    start_camera.style.display = "none"
    start_capture.style.display = "flex"
}

function error_func(error) {
    console.log("media access denied");
}

//to do
//this only captures a single image
start_capture.addEventListener('click',function(){
    //capture img from video
    captured_image.getContext('2d').drawImage(video,0,0,captured_image.width,captured_image.height);
    //get img data
    let imageData = captured_image.toDataURL("image/png")
    //convert img data into img source
    let imgStr = imageData.substr(22)
    //create new img file from img source
    let img = new Image();
    img.src = imgStr
    
    /*
    blah.src = imageData
    blah.width = blah.width< blah.height?blah.width:blah.height; 
    blah.height = blah.width< blah.height?blah.width:blah.height; 
    */

    //redraw image from the newly created file
    captured_image.getContext('2d').drawImage(img,0,0,canvas.width,canvas.height);
})

//decides whether to upload or wait for a certain time
//put this in the callback of the upload fetch request

document.getElementById("start").addEventListener("click",startContinuousUpload)
document.getElementById("stop").addEventListener("click",stopContinuousUpload)

//
function startContinuousUpload() {
    result.style.display = 'block';
    action_record = []
    actionCounter = 0
    accuracyRecord = []
    one_click_joint_record = []
    startingTime = Date.now()
    timestampList = []
    imgList = []
    stopUpload = false
    newImgTimer = setInterval(appendNewImage,IMAGE_INTERVAL * 1000,'File')
    setTimeout(newUpload,IMAGE_INTERVAL*IMAGE_BATCH*1000 + 100)
    setTimeout(continue_show_scene, IMAGE_INTERVAL*IMAGE_BATCH*1000 + 100)
}

function stopContinuousUpload() {
    result.style.display = 'none';
    time = null
    stopUpload = true
    clearInterval(newImgTimer)
    console.log(action_record)

}

function newUpload() {
    test_only()
}

// use static image to simulate real-time process 
function test_only() {
  let formData = new FormData()
  sliceImage(formData)
  //placeholder, delete later
  let timestamp =  Date.now()
  formData.append('action_choice', document.getElementById("Action_Choice").value)
  //console.log("current time: " + Date.now())
  //console.log("Starting time: " + startingTime)
  //console.log(`timestamp is ${timestamp}`)
  console.log (formData.keys())
  let data = {
    method: 'PUT',
    body: formData
  }
  let uploadTime = Date.now()
  fetch(url, data).then(response => {
    return response.json()
  })
    .then(res => {
      console.log(res)

      if (one_click_joint_record.length != 0) {
        let last_frame = one_click_joint_record[one_click_joint_record.length - 1]
        res.splice(0, 0, last_frame)
      }

         replay(res)
      


      //update_skeleton(res['smpl_joint_coords'], I2L_skeleton);
      for (let i = 0; i < res.length; i++) {
        if (res[i]['action_name'] != 'Loss exceeds threshold!') {
          action_record.push({
            'action_name': res[i]['action_name'],
            'loss': res[i]['loss']

          })
          // need to improve for a batch of image upload
          one_click_joint_record.push({
            'smpl_joint_coords': res[i]['smpl_joint_coords'],
            'timestamp': res[i]['timestamp']
          })
          accuracyRecord.push(res[i].action_accuracy)
        }
      }
      // action recognition
      let dict = new Object()
      let max_value = 0.00001
      let predicted_action = ''
      let total_loss = 0
      // need to extract data from res
      let replay_data = one_click_joint_record;

        if (action_record.length > 10) {

            for (let i = 2; i < action_record.length-2; i++) {
              if (action_record[i]['action_name'] in dict) {
                dict[action_record[i]['action_name']] += 1
                if (dict[action_record[i]['action_name']] > max_value) {
                  max_value = dict[action_record[i]['action_name']]
                  predicted_action = action_record[i]['action_name']
                }
              }
              else {
                dict[action_record[i]['action_name']] = 1
                  if (dict[action_record[i]['action_name']] > max_value) {
                  max_value = dict[action_record[i]['action_name']]
                  predicted_action = action_record[i]['action_name']
                }
              }
            }
          console.log ('predicted action is:'+ predicted_action)
          Action_Name.text = predicted_action

          for (let i = 0; i < action_record.length; i++) {
            if (action_record[i]['action_name'] == predicted_action)
              total_loss += action_record[i]['loss']
          }
          Average_Score.text = total_loss/max_value
          actionCounter++
      }
      if (!stopUpload){
        console.log("single upload time elapsed: " + (Date.now() - uploadTime)/1000)
        newUpload()
      }
        
      else {
        let replay_data = one_click_joint_record
        let tr = document.createElement("tr")
			  let td_1 = document.createElement("td")
			  td_1.appendChild(document.createTextNode(predicted_action))
			  let td_2 = document.createElement("td")
			  td_2.appendChild(document.createTextNode(total_loss/max_value))
			  let td_3 = document.createElement("td")
        td_3.appendChild(document.createTextNode('NA'))
        let td_4 = document.createElement("td");
			  let butt = document.createElement("button");
        td_4.appendChild(butt);
        butt.className = "btn btn-primary"
			  // use function closure to bind the block scpoed record_data to current one 
			  butt.addEventListener('click', ()=>{
				  // replay data should be in form as [{'smpl_joint_coords'}: [[x,y,z],[x,y,z], ...] , 'timestamp': } ...  ]
				  console.log(replay_data)
				  replay(replay_data)
			  });
			  tr.appendChild(td_1)
			  tr.appendChild(td_2)
        tr.appendChild(td_3)
        tr.appendChild(td_4);
        record.append(tr)
        Action_Name.text = ''
        Average_Score.text = ''
        console.log(accuracyRecord)
      }

    })
    .then(console.log("test succeed"))
    .catch(err => console.log(err))
}


//load src and convert to a File instance object
//work for any type of src, not only image src.
//return a promise that resolves with a File instance
function srcToFile(src, fileName, mimeType){
    return (fetch(src)
        .then(function(res){return res.arrayBuffer();})
        .then(function(buf){return new File([buf], fileName, {type:mimeType});})
    );
}

function dataURItoBlob(dataURI) {
    // convert base64 to raw binary data held in a string
    // doesn't handle URLEncoded DataURIs - see SO answer #6850276 for code that does this
    var byteString = atob(dataURI.split(',')[1]);

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    // write the bytes of the string to an ArrayBuffer
    var ab = new ArrayBuffer(byteString.length);
    var ia = new Uint8Array(ab);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    //New Code
    return new Blob([ab], {type: mimeString});

}




function captureAndUpload() {
    let imgFile = getNewImage('Capture')
    
    console.log(imgFile)
    let formData = new FormData()
    formData.append('image', imgFile)
    formData.append('action_choice', document.getElementById("Action_Choice").value)
    formData.append('timestamp',DEFAULT_TIMESTAMP)
    let data = {
          method: 'PUT',
          body: formData,
        }

      fetch(url,data).then(response => {
          return response.json()
        })
      .then( res => {
          console.log(res)

          // display result should be one_click_joint_record last element append res[smpl_joint_coords]
          update_skeleton(res['smpl_joint_coords'], I2L_skeleton);
          //update_skeleton(res['human36_joint_coords'], human36_skeleton);

          //one_click_joint_record append res[smpl_joint_coords]

        if (action_record[0]['action_name'] != 'Loss exceeds threshold!') {
          action_record.push({
            'action_name': res['action_name'],
            'loss': res['loss']
          })                
        }


          //update_skeleton(res['Sem_joints'], Sem_skeleton); 
        // action recognition
        let dict = new Object()
        let max_value = 0.00001
        let predicted_action = ''
        let total_loss = 0
        // need to extract data from res
        let replay_data = one_click_joint_record;
        if (action_record.length > 10) {

          for (let i = 2; i < action_record.length-2; i++) {
            if (action_record[i]['action_name'] in dict) {
              dict[action_record[i]['action_name']] += 1
              if (dict[action_record[i]['action_name']] > max_value) {
                max_value = dict[action_record[i]['action_name']]
                predicted_action = action_record[i]['action_name']
              }
            }
              else {
                dict[action_record[i]['action_name']] = 1
                if (dict[action_record[i]['action_name']] > max_value) {
                  max_value = dict[action_record[i]['action_name']]
                  predicted_action = action_record[i]['action_name']
                }
              }
            }
          console.log ('predicted action is:'+ predicted_action)
          Action_Name.text = predicted_action

          for (let i = 0; i < action_record.length; i++) {
            if (action_record[i]['action_name'] == predicted_action)
              total_loss += action_record[i]['loss']
          }
          Average_Score.text = total_loss/max_value
        }
          

        //check whether have decided to stop uploading
        if(!stopUpload){
          newUpload()
        }
        else {
          let tr = document.createElement("tr");
          let td_1 = document.createElement("td");
          td_1.appendChild(document.createTextNode(predicted_action));
          let td_2 = document.createElement("td");
          td_2.appendChild(document.createTextNode(total_loss / max_value));
          let td_3 = document.createElement("td");
          td_3.appendChild(document.createTextNode('NA'));
          let td_4 = document.createElement("td");
			    let butt = document.createElement("button");
          td_4.appendChild(butt);
          butt.className = "btn btn-primary"
			    // use function closure to bind the block scpoed record_data to current one 
			    butt.addEventListener('click', ()=>{
				  // replay data should be in form as [{'smpl_joint_coords'}: [[x,y,z],[x,y,z], ...] , 'timestamp': } ...  ]
				  console.log(replay_data)
				  //replay(replay_data)
			    });
          tr.appendChild(td_1);
          tr.appendChild(td_2);
          tr.appendChild(td_3);
          tr.appendChild(td_4);
          record.append(tr);
          Action_Name.text = ''
          Average_Score.text = ''
      }
      })
      .then(console.log("succeed here!"))
      .catch(error => console.log(error))
}

//a more direct way to slice image is to directly append the image to fordata
const sliceImage = (formData) => {
  //copy the imgList
  let uploadImgList = [...imgList]
  let uploadTimestampList = [...timestampList]
  //clear the orignial imgList so it doesn't affect the img capturing
  imgList = []
  timestampList = []
  //the slice interval is larger than 1 due to the setting of IMAGE_BATCH AND INTERVAL
  let sliceInterval = uploadImgList.length / IMAGE_BATCH
  //append img to the formData
  console.log(`captured ${uploadImgList.length} images and ${uploadTimestampList.length} timestamps between uploads`)
  for(let i=0; i < IMAGE_BATCH; i++){
    formData.append('image',uploadImgList[Math.min(Math.round(i * sliceInterval),uploadImgList.length-1)])
    formData.append('timestamp',uploadTimestampList[Math.min(Math.round(i * sliceInterval),uploadTimestampList.length-1)])
  }
  return formData
}

const getNewImage = (source) => {
  let imgFile
  if(source == 'File'){
    imgFile = document.getElementById("form-image").files[0]
  }else if(source == 'Capture'){
    //capture img from video
    captured_image.getContext('2d').drawImage(video,0,0,captured_image.width,captured_image.height);
    //get img data
    let imageData = captured_image.toDataURL("image/png")
    //convert img data into img source
    let imgStr = imageData.substr(22)
    //create new img file from img source
    let img = new Image();
    img.src = imgStr
    //redraw image from the newly created file
    captured_image.getContext('2d').drawImage(img,0,0,canvas.width,canvas.height);

    let imgBlob = dataURItoBlob(imageData)
    imgFile = new File([imgBlob], "image.png");
  }

  return imgFile
}
//append a new image to the pending upload list
//param source identifies where the image comes from
const appendNewImage = (source) => {
    let img = getNewImage(source)
    imgList.push(img)
    timestampList.push(Date.now()-startingTime)
}

