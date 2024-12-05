import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import '../styles/PreprocessData.css';
import L from 'leaflet';

const customIcon = L.icon({
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  iconSize: [25, 41], // Size of the icon
  iconAnchor: [12, 41], // Point of the icon which will correspond to marker's location
  popupAnchor: [1, -34], // Point from which the popup should open relative to the iconAnchor
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  shadowSize: [41, 41] // Size of the shadow
});

const PreprocessData = () => {
  const [position, setPosition] = useState(null);
  const [startDate, setStartDate] = useState(new Date());
  const [endDate, setEndDate] = useState(new Date());
  const [confirmationMessage, setConfirmationMessage] = useState('');

  const handleMapClick = (latlng) => {
    setPosition({
      lat: latlng.lat.toFixed(2),
      lng: latlng.lng.toFixed(2)
    });
  };

  const handlePreprocessData = async () => {
    if (!position) {
      alert('Please click on the map to select a location first.');
      return;
    }
    try {
      const latitude = position.lat.toString();
      const longitude = position.lng.toString();
      console.log(latitude, longitude, startDate.toISOString().split('T')[0], endDate.toISOString().split('T')[0]);
      const response = await axios.post('http://localhost:8100/preprocessing', {
        latitude,
        longitude,
        start_date: startDate.toISOString().split('T')[0],
        end_date: endDate.toISOString().split('T')[0]
      });
      console.log(response.data);
      setConfirmationMessage('Data preprocessed and saved to the database.');
    } catch (error) {
      console.error('Error preprocessing data:', error);
      setConfirmationMessage('Error preprocessing data.');
    }
  };

  const LocationMarker = () => {
    useMapEvents({
      click(e) {
        handleMapClick(e.latlng);
      },
    });

    return position === null ? null : (
      <Marker position={position} icon={customIcon}>
        <Popup>
          Coordinates: {position.lat}, {position.lng}
        </Popup>
      </Marker>
    );
  };

  return (
    <div>
      <h1>Click on the map to get coordinates</h1>
      <div className="date-picker-container">
        <div className="date-picker">
          <label>Start Date: </label>
          <DatePicker selected={startDate} onChange={(date) => setStartDate(date)} />
        </div>
        <div className="date-picker">
          <label>End Date: </label>
          <DatePicker selected={endDate} onChange={(date) => setEndDate(date)} />
        </div>
      </div>
      <div className="map-container">
        <MapContainer center={[41.3888, 2.159]} zoom={13} style={{ height: '100%', width: '50%' }}>
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          <LocationMarker />
        </MapContainer>
      </div>
      <button className="round-pill-button" onClick={handlePreprocessData}>Preprocess Data</button>
      {confirmationMessage && <div className="confirmation-message">{confirmationMessage}</div>}
    </div>
  );
};

export default PreprocessData;