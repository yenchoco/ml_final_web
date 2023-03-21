import React, {useState, useEffect}from 'react';
import Typography from "@mui/material/Typography";

function App(){
    const [flag,setFlag] = useState('not yet uploaded');
    // const [show,setShow] = useState([])
  // const [data,setData] = useState([{}]);
//   const [LogoFile, setLogoFile] = useState();

//   const selectLogoFile = (event) => {
//     setLogoFile(URL.createObjectURL(event.target.files[0]));
// };

  // useEffect(() => {
  //   fetch("/members").then(
  //     res => res.json()
  //   ).then(
  //     data => {
  //       setData(data)
  //       console.log(data)
  //     }
  //   )
  // },[])

  const handleSubmit = (e) => {
    e.preventDefault()
    const formData = new FormData(e.target);
    
    const Upload = async() => {
      await fetch('/api/upload', {
        method: 'POST',
        body: formData
      }).then(resp => {
        resp.json().then(
          idc => {
            if(idc.val==0)
              setFlag('homo')
            else if(idc.val==1)
              setFlag('hetero')
            console.log(idc.val)})
      })
    }
    Upload();
  }

  return (
    <form onSubmit={handleSubmit} className="container mt-5 pt-5 pb-5" enctype="multipart/form-data">
    <div className="form-inline justify-content-center mt-5">
        <h1 htmlFor="image" className="ml-sm-4 font-weight-bold mr-md-4">
        Homo-sexual Classifier 
        </h1>
        <p>
          This is a gay classifier for asia males, you can upload photo with that in the image.
          <br/>
          As this model got 75% of accuracy rate among this group of people, please don't take the result too seriously.
          <br/>
          Have fun :D
          <br/><br/>
          **********************notice**********************
          <br/>
          the photo must contain only one face in it!!!
        </p>
        <div className="input-group">
            <input type="file" id="image" name="file" 
            accept="image/*" className="file-custom"/>
        </div>
    </div>
    <div className="input-group justify-content-center mt-4">
        <button type="submit" className="btn btn-md btn-primary">Upload</button>
    </div>
    <div>
      result : {flag}
    </div>
  </form>
  )
}

export default App