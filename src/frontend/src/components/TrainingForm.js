import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { MapContainer, TileLayer, Marker } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import '../styles/Chart.css';

const customIcon = L.icon({
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  shadowSize: [41, 41]
});

const TrainModelForm = () => {
  const [latitude, setLatitude] = useState("41.389");
  const [longitude, setLongitude] = useState("2.159");
  const [startDate, setStartDate] = useState(new Date('2024-09-01'));
  const [endDate, setEndDate] = useState(new Date('2024-10-02'));
  const [coordinatePairs, setCoordinatePairs] = useState([]);
  const [result, setResult] = useState(null);
  const mapRef = useRef();

  useEffect(() => {
    axios.get('http://localhost:8100/processing/coordinates')
      .then(response => {
        setCoordinatePairs(response.data.coordinates);
        if (response.data.coordinates.length > 0) {
          setLatitude(response.data.coordinates[0].latitude);
          setLongitude(response.data.coordinates[0].longitude);
        }
      })
      .catch(error => {
        console.error('There was an error fetching the coordinates!', error);
      });
  }, []);

  useEffect(() => {
    if (mapRef.current) {
      mapRef.current.flyTo([latitude, longitude], 13);
    }
  }, [latitude, longitude]);

  const handleMapClick = (e) => {
    setLatitude(e.latlng.lat.toString());
    setLongitude(e.latlng.lng.toString());
  };

  const handleLatitudeChange = (event) => {
    const selectedLatitude = event.target.value;
    const selectedPair = coordinatePairs.find(pair => pair.latitude === selectedLatitude);
    if (selectedPair) {
      setLatitude(selectedLatitude);
      setLongitude(selectedPair.longitude);
    } else {
      setLatitude(selectedLatitude);
    }
  };

  const handleLongitudeChange = (event) => {
    const selectedLongitude = event.target.value;
    const selectedPair = coordinatePairs.find(pair => pair.longitude === selectedLongitude);
    if (selectedPair) {
      setLongitude(selectedLongitude);
      setLatitude(selectedPair.latitude);
    } else {
      setLongitude(selectedLongitude);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const formattedStartDate = formatDate(startDate);
    const formattedEndDate = formatDate(endDate);

    const payload = {
      latitude: latitude.toString(),
      longitude: longitude.toString(),
      start_date: formattedStartDate,
      end_date: formattedEndDate
    };

    console.log("Payload:", payload);

    try {
      const response = await fetch('http://localhost:8100/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const responseText = await response.text();
      console.log("Raw response:", responseText);

      if (!response.ok) {
        console.error('Error response:', responseText);
      } else {
        const data = JSON.parse(responseText);
        setResult(data);
      }
    } catch (error) {
      console.error('Error submitting the form:', error);
    }
  };

  const formatDate = (date) => {
    const d = new Date(date);
    let month = '' + (d.getMonth() + 1);
    let day = '' + d.getDate();
    const year = d.getFullYear();
    let hour = '' + d.getHours();
    let minute = '' + d.getMinutes();

    if (month.length < 2) month = '0' + month;
    if (day.length < 2) day = '0' + day;
    if (hour.length < 2) hour = '0' + hour;
    if (minute.length < 2) minute = '0' + minute;

    return [year, month, day].join('-') + ' ' + [hour, minute].join(':');
  };

  return (
    <div>
      <h2>Train Model</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group date-picker-row">
          <div>
            <label>Start Date: </label>
            <DatePicker 
              selected={startDate} 
              onChange={date => setStartDate(date)} 
              showTimeSelect
              dateFormat="Pp"
              popperClassName="datepicker-container"
            />
          </div>
          <div>
            <label>End Date: </label>
            <DatePicker 
              selected={endDate} 
              onChange={date => setEndDate(date)} 
              showTimeSelect
              dateFormat="Pp"
              popperClassName="datepicker-container"
            />
          </div>
        </div>
        <div className="form-group coordinate-row">
          <div>
            <label>Latitude:</label>
            <select value={latitude} onChange={handleLatitudeChange}>
              {coordinatePairs.map(pair => (
                <option key={pair.latitude} value={pair.latitude}>
                  {pair.latitude}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Longitude:</label>
            <select value={longitude} onChange={handleLongitudeChange}>
              {coordinatePairs.map(pair => (
                <option key={pair.longitude} value={pair.longitude}>
                  {pair.longitude}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="map-container">
          <MapContainer center={[parseFloat(latitude), parseFloat(longitude)]} zoom={13} style={{ height: "200px", width: "100%" }} onClick={handleMapClick} ref={mapRef}>
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <Marker position={[parseFloat(latitude), parseFloat(longitude)]} icon={customIcon} />
          </MapContainer>
        </div>
        <button type="submit">Train</button>
      </form>
      {result && <div>{result.message} - Execution Time: {result.execution_time}s</div>}
    </div>
  );
};

export default TrainModelForm;