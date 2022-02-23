import * as THREE from 'https://cdn.skypack.dev/three@0.133.1';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.133.1/examples/jsm/controls/OrbitControls.js';
export { update_skeleton}

document.getElementById("form-image").onchange = evt => {
  let image = document.getElementById("form-image").files[0]
  if (image) {
    blah.src = URL.createObjectURL(image)
    blah.width = blah.width< blah.height?blah.width:blah.height; 
    blah.height = blah.width< blah.height?blah.width:blah.height; 
  }
}
form.addEventListener("submit",function(event){

    event.preventDefault();
    window.alert("trying to upload file to" + ngrok_url)

    let file = document.getElementById("form-image").files[0]
    console.log(file)
    if (file == null || file == ''){
        window.alert("You haven't choose a file yet")
        return
    }
    
    let url = "";
    let formData = new FormData()
    
    if (file.type == 'video/mp4'){
	      url = `${ngrok_url}action_upload`
    	  formData.append('video', file)
	
	      let data = {
        method: 'POST',
        body: formData
    	  }
    
        fetch(url, data)
        .then(response => response.json())
        .then(
            data => {
            console.log(data);
            update_action_list()
          })
        .catch(error => console.log(error))	
    }
	
	
    else{
	      url = `${ngrok_url}imageUpload`
    	  formData.append('image',file)
    
    
    	  let data = {
        method: 'POST',
        body: formData
    	  }
    
        fetch(url, data).then(response => response.json())
        .then(
            data => {
            console.log(data);
            //update_skeleton(data['smpl_joint_coords'], I2L_skeleton);
            //update_skeleton(data['human36_joint_coords'], human36_skeleton);
            //update_skeleton(data['Sem_joints'], Sem_skeleton); 
	    	    if (data['action_name']== 'Loss exceeds threshold!')
		            alert("No matching action found!")
	          else  	
		            document.getElementById("Action_Choice").value = data['action_name'] 
                })
        .catch(error => console.log(error))
    }
})


function update_action_list(){
	let old_value = new Set()
	for (let i = 0; i<  Action_Choice.children.length; i++){
		old_value.add(Action_Choice[i].value)
	}
	
	let url = `${ngrok_url}getFitness`
	let data = {method: 'GET'}
	fetch(url, data)
	.then(res=> res.json())
	.then(new_action=>{
	for (let i=0; i<new_action.length; i++){
		if (!( old_value.has(new_action[i])) ){
			//console.log(new_action[i])
			let new_option = document.createElement('option')
			new_option.value = new_action[i]
			new_option.text = new_action[i]
			Action_Choice.append(new_option)
		}
		}
	})
}
update_action_list()




let camera, controls, scene, renderer

// 2d ARRAY
function flatten_array(multi_dim_array){
    let  oneD_position = [];
    /*for(let i = 0; i < multi_dim_array[0].length; i++)
    {
        oneD_position = oneD_position.concat(multi_dim_array[0][i]);
    }*/
    for(let i = 0; i < multi_dim_array.length; i++)
    {
        oneD_position = oneD_position.concat(multi_dim_array[i]);
    }
    return oneD_position;
}

function update_skeleton(new_position,skeleton){
    //set Pelvis to the origin
    new_position = flatten_array(new_position);
    console.log(new_position);
    let Pelvis_index = 0;
    
    let Pelvis_x = new_position[3*Pelvis_index];
    let Pelvis_y = new_position[3*Pelvis_index+1];
    let Pelvis_z = new_position[3*Pelvis_index+2];
    for (let i=0; i<new_position.length; i+=3){

        new_position[i] -=  Pelvis_x;
        new_position[i+1] -=  Pelvis_y;
        new_position[i+2] -=  Pelvis_z;
    }   

    const geometry = skeleton.geometry;    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(new_position,3));
    
    }


function create_skelton(position, indices){
    //set Pelvis to the origin
  if (position.length == 0) {

    const geometry = new THREE.BufferGeometry();    
    geometry.setIndex(indices);    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(position,3));
    
    const material = new THREE.MeshBasicMaterial( { color: 0xFF69B4 } );
    let skeleton = new THREE.LineSegments(geometry, material);
    const quaternion = new THREE.Quaternion();
    //quaternion.setFromUnitVectors( spine_vecotor,new THREE.Vector3( 0, 1, 0 ) );
    quaternion.setFromAxisAngle( new THREE.Vector3( 1, 0, 0 ), Math.PI  );
    skeleton.applyQuaternion(quaternion);
    return skeleton;

    
  }
  
    let Pelvis_index = 0, chest_index = 3;

    let Pelvis_x = position[3 * Pelvis_index];
    let Pelvis_y = position[3 * Pelvis_index + 1];
    let Pelvis_z = position[3 * Pelvis_index + 2];
    for (let i = 0; i < position.length; i += 3) {

      position[i] -= Pelvis_x;
      position[i + 1] -= Pelvis_y;
      position[i + 2] -= Pelvis_z;
    }
  

    const geometry = new THREE.BufferGeometry();    
    geometry.setIndex(indices);    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(position,3));
    
    const material = new THREE.MeshBasicMaterial( { color: 0xFF69B4 } );
    let skeleton = new THREE.LineSegments(geometry, material);
    
    let spine_vecotor = new THREE.Vector3( position[chest_index*3]-position[Pelvis_index*3], position[chest_index*3+1]-position[Pelvis_index*3+1], position[chest_index*3+2]-position[Pelvis_index*3+2] ).normalize();
    const quaternion = new THREE.Quaternion();
    //quaternion.setFromUnitVectors( spine_vecotor,new THREE.Vector3( 0, 1, 0 ) );
    quaternion.setFromAxisAngle( new THREE.Vector3( 1, 0, 0 ), Math.PI  );
    skeleton.applyQuaternion(quaternion);
    
    return skeleton;
    }
    


function init(){

    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0xeeeeee );
    scene.add( new THREE.GridHelper( 400, 10 ) );
    scene.add( new THREE.AxesHelper( 100 ) );

    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.set( 15, 20, 30 );

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize( canvas.offsetWidth - 25, window.innerHeight - 60  > 720 ? window.innerHeight - 60  : 720 );

    canvas.setAttribute("height",window.innerHeight - 60 > 720 ? window.innerHeight - 60 : 720)
    canvas.setAttribute("width", window.innerWidth - 20)
    canvas.appendChild( renderer.domElement );

    controls = new OrbitControls( camera, renderer.domElement );
    window.addEventListener( 'resize', onWindowResize );

  I2L_skeleton = create_skelton(init_position, I2L_indices);
    scene.add(I2L_skeleton);
  human36_skeleton = create_skelton([], human36_indices);
  human36_skeleton.position.x = -40;
    scene.add(human36_skeleton);
  Sem_skeleton = create_skelton([], human36_indices);
  Sem_skeleton.position.x = 40;
    scene.add(Sem_skeleton);
}


function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    canvas.setAttribute("height",window.innerHeight - 60  > 720 ? window.innerHeight - 60  : 720)
    
    renderer.setSize( canvas.offsetWidth - 6, window.innerHeight - 60  > 720 ? window.innerHeight - 60  : 720 );

}


function animate() {
	requestAnimationFrame( animate );

	renderer.render( scene, camera );
}



init();
animate();
