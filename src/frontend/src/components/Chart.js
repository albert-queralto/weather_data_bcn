import React, { useEffect, useState } from 'react';
import axios from 'axios';
import * as am5 from "@amcharts/amcharts5";
import * as am5xy from "@amcharts/amcharts5/xy";
import am5themes_Animated from "@amcharts/amcharts5/themes/Animated";
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import '../styles/Chart.css'; // Import the CSS file

const ChartComponent = () => {
  const [data, setData] = useState([]);
  const [latitude, setLatitude] = useState("41.389");
  const [longitude, setLongitude] = useState("2.159");
  const [startDate, setStartDate] = useState(new Date('2024-10-01'));
  const [endDate, setEndDate] = useState(new Date('2024-10-02'));

  // curl -X POST -H "Content-Type: application/json" -d '{"latitude":"41.389", "longitude":"2.159", "start_date":"2024-10-01", "end_date":"2024-10-02"}' http://localhost:8100/get_data

  useEffect(() => {
    axios.post('http://localhost:8100/get_data', {
      latitude,
      longitude,
      start_date: startDate.toISOString().split('T')[0],
      end_date: endDate.toISOString().split('T')[0]
    })
    .then(response => {
      setData(response.data.data);
    })
    .catch(error => {
      console.error('There was an error fetching the data!', error);
    });
  }, [latitude, longitude, startDate, endDate]);

  useEffect(() => {
    let root = am5.Root.new("chartdiv");
    root.setThemes([am5themes_Animated.new(root)]);

    let chart = root.container.children.push(am5xy.XYChart.new(root, {
      panX: true,
      panY: true,
      wheelX: "panX",
      wheelY: "zoomX",
      pinchZoomX: true
    }));

    let xAxis = chart.xAxes.push(am5xy.DateAxis.new(root, {
      maxDeviation: 0.2,
      baseInterval: {
        timeUnit: "day",
        count: 1
      },
      renderer: am5xy.AxisRendererX.new(root, {}),
      tooltip: am5.Tooltip.new(root, {})
    }));

    let yAxis = chart.yAxes.push(am5xy.ValueAxis.new(root, {
      renderer: am5xy.AxisRendererY.new(root, {})
    }));

    const variableCodes = [...new Set(data.map(item => item.variable_code))];

    variableCodes.forEach(variable_code => {
      let series = chart.series.push(am5xy.LineSeries.new(root, {
        name: variable_code,
        xAxis: xAxis,
        yAxis: yAxis,
        valueYField: "value",
        valueXField: "timestamp",
        tooltip: am5.Tooltip.new(root, {
          labelText: "{valueY}"
        })
      }));

      series.data.setAll(data.filter(item => item.variable_code === variable_code));
    });

    chart.set("cursor", am5xy.XYCursor.new(root, {}));
    chart.set("scrollbarX", am5.Scrollbar.new(root, {
      orientation: "horizontal"
    }));

    return () => {
      root.dispose();
    };
  }, [data]);

  const handleMapClick = (e) => {
    setLatitude(e.latlng.lat);
    setLongitude(e.latlng.lng);
  };

  return (
    <div className="container">
      <div className="sidebar">
        <label>Start Date: </label>
        <DatePicker selected={startDate} onChange={date => setStartDate(date)} />
        <label>End Date: </label>
        <DatePicker selected={endDate} onChange={date => setEndDate(date)} />
        {/* <div className="map-container">
          <MapContainer center={[latitude, longitude]} zoom={13} style={{ height: "200px", width: "100%" }} onClick={handleMapClick}>
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
          </MapContainer>
        </div> */}
      </div>
      <div className="main-content">
        <div className="chart-container">
          <div id="chartdiv" style={{ width: "100%", height: "400px" }}></div>
        </div>
      </div>
    </div>
  );
};

export default ChartComponent;