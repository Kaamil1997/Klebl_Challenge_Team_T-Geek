import React, { useState,useRef,useEffect } from "react";
import data from "./data_model.json";
import { Table } from "antd";
import "./App.css";
import ReactPDF from '@react-pdf/renderer';
import Popup from 'reactjs-popup';
import 'reactjs-popup/dist/index.css';


const columns = [
  { title: "Pos", dataIndex: "Pos", key: "Pos" },
  { title: "Anz.", dataIndex: "Anz.", key: "Anz." },
  { title: "⌀", dataIndex: "⌀", key: "⌀" },
  { title: "Lange", dataIndex: "lange", key: "lange" },
  { title: "Bem.", dataIndex: "Bem.", key: "Bem." },
];




const App = () => {
  
  const [isUploaded, setIsUploaded] = useState(false);
  const [currentView, setCurrentView] = useState("form");
  const [formData, setFormData] = useState(data);
  const [test, setTest] = useState(false);

  const handleInputChange = (e, section, index, field) => {
    const updatedData = { ...formData };
    updatedData[section][index][field] = e.target.value;
    setFormData(updatedData);
  };

  const canvasRef = useRef(null);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return; // Ensure canvas exists
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
      ctx.beginPath();
      ctx.rect(380, 470, 150, 10); // Draw a rectangle
      ctx.strokeStyle = 'red'; // Set outline color
      ctx.lineWidth = 2; // Set the stroke width
      ctx.stroke(); // Render the rectangle outline
      
    }
  };

  const tester = () => {
    setTest(true)
  }

  useEffect(() => {
    if (test){
      draw();
    }
     // Call the draw function after the canvas is mounted
  }, [test]);

  const isNested = (section, index) => {
    return (
      Array.isArray(formData[section]) &&
      typeof formData[section][index] === "object"
    );
  };

  const renderForm = () => (
    <>
      <div className="container2">      
      <div>
      {/* Render Plankopf */}
      <h3 className="h3">Plankopf</h3>
      {formData.Plankopf.map((item, index) => (
        <div className="subcontainer" key={index}>
          {Object.keys(item).map((key) => (
            <div className="input-group" key={key}>
              <label onClick={tester}>{key}:</label>
              <input
                type="text"
                value={item[key]}
                onChange={(e) => handleInputChange(e, "Plankopf", index, key)}
              />
            </div>
          ))}
        </div>
      ))}

      {/* Render List Einbauteile */}
      <h3 className="h3">List Einbauteile</h3>
      {formData.list_einbauteile.map((item, index) => (
        <div className="subcontainer" key={index}>
          {Object.keys(item).map((key) => (
            <div key={key}>
              <label>{key}:</label>
              <input
                type="number"
                value={item[key]}
                onChange={(e) =>
                  handleInputChange(e, "list_einbauteile", index, key)
                }
              />
            </div>
          ))}
        </div>
      ))}

      {/* Render Liste Stahl */}
      <h3 className="h3">Liste Stahl</h3>
      {Object.keys(formData.liste_Stahl).map((key) => (
        <div className="subcontainer" key={key}>
          <h4>{key}</h4>
          {formData.liste_Stahl[key].map((item, index) => (
            <div className="subcontainer" key={index}>
              {Object.keys(item).map((field) => (
                <div key={field}>
                  <label>{field}:</label>
                  <input
                    type="text"
                    value={item[field]}
                    onChange={(e) =>
                      handleInputChange(e, "liste_Stahl", index, field)
                    }
                  />
                </div>
              ))}
            </div>
          ))}
        </div>
      ))}

      {/* Render Vorderansicht */}
      <h3 className="h3">Vorderansicht</h3>
      {Object.keys(formData.Vorderansicht).map((key) => (
        <div className="subcontainer" key={key}>
          <h4>{key}</h4>
          {Object.keys(formData.Vorderansicht[key]).map((subKey) => (
            <div className="subcontainer" key={subKey}>
              <label>{subKey}:</label>
              <input
                type="number"
                value={formData.Vorderansicht[key][subKey]}
                onChange={(e) =>
                  handleInputChange(e, "Vorderansicht", key, subKey)
                }
              />
            </div>
          ))}
        </div>
      ))}

      {/* Render Seitenansicht/Draufsicht */}
      <h3 className="h3">Seitenansicht/Draufsicht</h3>
      {Object.keys(formData["Seitenansischt/Draufsicht"]).map((key) => (
        <div className="subcontainer" key={key}>
          <h4>{key}</h4>
          {Object.keys(formData["Seitenansischt/Draufsicht"][key]).map(
            (subKey) => (
              <div className="subcontainer" key={subKey}>
                <label>{subKey}:</label>
                <input
                  type="number"
                  value={formData["Seitenansischt/Draufsicht"][key][subKey]}
                  onChange={(e) =>
                    handleInputChange(
                      e,
                      "Seitenansicht/Draufsicht",
                      key,
                      subKey
                    )
                  }
                />
              </div>
            )
          )}
        </div>
      ))} 
      </div>    
      <div>
      {
        test?<canvas
        ref={canvasRef}
        width={600}
        height={500}
        style={{ border: '0px solid red',position: 'absolute' }}
      ></canvas>:<></>
      }
      
        <iframe src="/file.pdf" style={{ width: "100%", height: "500px" }} frameborder="0"></iframe>        
        <Popup trigger={<button>Verify Json</button>} position="top center">
    <div>Json Verified ✅</div>
  </Popup>
        
      </div>   
      </div>
      {/* Add more sections as necessary */}
    </>
  );
  const getColumns = (section) => {
    if (Array.isArray(formData[section])) {
      const columns = Object.keys(formData[section][0]).map((key) => ({
        title: key,
        dataIndex: key,
        key: key,
      }));
      console.log("🚀 ~ columns ~ columns:", columns);
      return Object.keys(formData[section][0]).map((key) => ({
        title: key,
        dataIndex: key,
        key: key,
      }));
    } else if (typeof formData[section] === "object") {
      return Object.keys(formData[section]).map((key) => ({
        title: key,
        dataIndex: key,
        key: key,
      }));
    }
    return [];
  };

  // Render table view based on the JSON structure
  const renderTable = () => (
    <div>
      <h3 className="h3">Plankopf</h3>
      <Table
        dataSource={formData.Plankopf}
        columns={getColumns("Plankopf")}
        rowKey={(record, index) => index.toString()}
        rowClassName={(record, index) =>
          isNested("Plankopf", index) ? "nested-row" : ""
        }
      />

      <h3 className="h3">List Einbauteile</h3>
      <Table
        dataSource={formData.list_einbauteile}
        columns={getColumns("list_einbauteile")}
        rowKey={(record, index) => index.toString()}
        rowClassName={(record, index) =>
          isNested("list_einbauteile", index) ? "nested-row" : ""
        }
      />

      <h3 className="h3">Liste Stahl</h3>
      {Object.keys(formData.liste_Stahl).map((key, index) => {
        return (
          <div className="subcontainer" key={key}>
            <h4>{key}</h4>
            <Table
              dataSource={formData.liste_Stahl[key]} // Correctly pass the data for each key
              columns={columns} // Dynamically get the columns
              rowKey={(record, index) => index.toString()}
              rowClassName={(record, index) =>
                isNested("liste_Stahl", index) ? "nested-row" : ""
              }
            />
          </div>
        );
      })}

      <h3 className="h3">Vorderansicht</h3>
      {Object.keys(formData.Vorderansicht).map((key) => (
        <div className="subcontainer" key={key}>
          <h4>{key}</h4>
          <Table
            dataSource={[formData.Vorderansicht[key]]}
            columns={Object.keys(formData.Vorderansicht[key]).map((field) => ({
              title: field,
              dataIndex: field,
              key: field,
            }))}
            rowKey={(record, index) => index.toString()}
            rowClassName={(record, index) =>
              isNested("Vorderansicht", index) ? "nested-row" : ""
            }
          />
        </div>
      ))}

      <h3 className="h3">Seitenansicht/Draufsicht</h3>
      {Object.keys(formData["Seitenansischt/Draufsicht"]).map((key) => (
        <div className="subcontainer" key={key}>
          <h4>{key}</h4>
          <Table
            dataSource={[formData["Seitenansischt/Draufsicht"][key]]}
            columns={Object.keys(
              formData["Seitenansischt/Draufsicht"][key]
            ).map((field) => ({
              title: field,
              dataIndex: field,
              key: field,
            }))}
            rowKey={(record, index) => index.toString()}
            rowClassName={(record, index) =>
              isNested("Seitenansicht/Draufsicht", index) ? "nested-row" : ""
            }
          />
        </div>
      ))}
    </div>
  );

  const renderJson = () => <pre>{JSON.stringify(formData, null, 2)}</pre>;
  const [loading, setLoading] = useState(false);

  const handleFileUpload = (e) => {
    setLoading(true);
    setTimeout(() => {
      setIsUploaded(true);
      setLoading(false);
    }, 5000);
  };

  return (

    <div className="container">
      {!isUploaded ? (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            textAlign: "center",
          }}
        >
          {loading ? (
            <p>Loading...</p>
          ) : isUploaded ? (
            <p>File Uploaded Successfully!</p>
          ) : (
            <div>
              <input type="file" onChange={handleFileUpload} />
              <p>Choose a file to upload</p>
            </div>
          )}
        </div>
      ) : (
        <>
          <div className="button-container">
            <button onClick={() => setCurrentView("form")}>Form View</button>
            <button onClick={() => setCurrentView("table")}>Table View</button>
            <button onClick={() => setCurrentView("json")}>JSON View</button>
          </div>
          {currentView === "form" && renderForm()}
          {currentView === "table" && renderTable()}
          {currentView === "json" && renderJson()}
        </>
      )}
    </div>
  );
};

export default App;
