let media = document.querySelector('#video-player')


const startVideoPlayback = () => {
    console.log("playing media")
    media.play()
}

const stopVideoPlayback = () => {
    if(!media.paused) {
        console.log("passing video")
        media.pause()
        media.currentTime = 0;
    }
}

const changeVideoSource = (event) => {
    console.log(`public/Fitness_video/${event.target.value}.mp4`)
    media.setAttribute('src',`public/Fitness_video/${event.target.value}.mp4`)
    media.load()
}

document.getElementById("start").addEventListener("click",startVideoPlayback)
document.getElementById("stop").addEventListener("click",stopVideoPlayback)
document.getElementById("Action_Choice").addEventListener("change",changeVideoSource)

